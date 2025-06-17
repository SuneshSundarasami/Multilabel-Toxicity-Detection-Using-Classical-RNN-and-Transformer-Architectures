import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import yaml
import os
from collections import Counter
from tqdm import tqdm

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'locations.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ToxicDataset(Dataset):
    """Dataset class for toxic comment classification"""
    def __init__(self, tokens, labels=None):
        self.tokens = tokens
        self.labels = labels
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = torch.LongTensor(self.tokens[idx])
        if self.labels is not None:
            labels = torch.FloatTensor(self.labels[idx])
            return tokens, labels
        return tokens

def collate_batch(batch):
    """Collate function for DataLoader - pads sequences"""
    if len(batch[0]) == 2:  # Has labels
        tokens, labels = zip(*batch)
        tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return tokens, labels
    else:  # No labels (test data)
        tokens = pad_sequence(batch, batch_first=True, padding_value=0)
        return tokens

def load_glove_embeddings(glove_path, word2idx, embedding_dim):
    """Load GloVe embeddings and return as tensor"""
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    embeddings_index = {}
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            try:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                
                if len(vector) == embedding_dim:
                    embeddings_index[word] = vector
                    
            except (ValueError, IndexError):
                continue
    
    print(f"Loaded {len(embeddings_index)} word vectors")
    
    # Create embedding matrix
    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    found_words = 0
    
    for word, idx in word2idx.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
            found_words += 1
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    print(f"Found embeddings for {found_words}/{len(word2idx)} words")
    
    # Return as tensor instead of numpy array
    return torch.FloatTensor(embedding_matrix)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
            
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, class_gammas=None):
        super().__init__()
        self.alpha = alpha
        self.class_gammas = class_gammas
        self.epsilon = 1e-6
        
    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, self.epsilon, 1 - self.epsilon)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Different gamma for each class
        focal_loss = torch.zeros_like(bce_loss)
        class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        for i, col in enumerate(class_names):
            gamma = self.class_gammas.get(col, 2.0)  # Default gamma is 2.0
            pt = torch.exp(-bce_loss[:, i])
            focal_weight = (1 - pt) ** gamma
            
            # Apply class weights if provided
            if self.alpha is not None:
                focal_weight = focal_weight * self.alpha[i]
                
            focal_loss[:, i] = focal_weight * bce_loss[:, i]
        
        return focal_loss.mean()

# Model 1: Simple GRU
class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=1, dropout=0.3, pretrained_embeddings=None):
        super(SimpleGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
            
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        gru_out, hidden = self.gru(embedded)
        # Use last hidden state
        output = self.fc(self.dropout(hidden[-1]))
        return torch.sigmoid(output)

# Model 2: LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state
        output = self.fc(self.dropout(hidden[-1]))
        return torch.sigmoid(output)

# Model 3: BiLSTM with Attention
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super(BiLSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           bidirectional=True, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        output = self.fc(self.dropout(context_vector))
        return torch.sigmoid(output)

def create_vocabulary(tokens_list, min_freq=2):
    """Create vocabulary from tokenized texts"""
    word_counts = Counter()
    for tokens in tokens_list:
        word_counts.update(tokens)
    
    # Create word to index mapping
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
    
    return word2idx, idx2word

def tokens_to_indices(tokens_list, word2idx):
    """Convert token lists to index lists"""
    indices_list = []
    for tokens in tokens_list:
        indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
        indices_list.append(indices)
    return indices_list

def oversample_minority_classes(tokens, labels, multipliers=None):
    """Oversample minority classes"""
    if multipliers is None:
        multipliers = {'threat': 3, 'identity_hate': 2, 'severe_toxic': 2}
    
    labels_df = labels.copy()
    tokens_list = tokens.copy()
    
    # Get column names
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    augmented_tokens = []
    augmented_labels = []
    
    for col_idx, col in enumerate(columns):
        if col in multipliers:
            # Find positive samples for this class
            pos_indices = [i for i, label in enumerate(labels) if label[col_idx] == 1]
            
            # Duplicate positive samples
            for _ in range(multipliers[col] - 1):
                for idx in pos_indices:
                    augmented_tokens.append(tokens_list[idx])
                    augmented_labels.append(labels[idx])
    
    # Combine original and augmented data
    final_tokens = tokens_list + augmented_tokens
    final_labels = np.vstack([labels, np.array(augmented_labels)]) if augmented_labels else labels
    
    return final_tokens, final_labels

def get_model_by_name(model_name, vocab_size, embedding_dim, hidden_dim, output_dim, 
                     pretrained_embeddings=None):
    """Factory function to create models by name"""
    if model_name.lower() == 'gru':
        return SimpleGRU(vocab_size, embedding_dim, hidden_dim, output_dim, 
                        pretrained_embeddings=pretrained_embeddings)
    elif model_name.lower() == 'lstm':
        return LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 
                             pretrained_embeddings=pretrained_embeddings)
    elif model_name.lower() == 'bilstm_attention':
        return BiLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, 
                                  pretrained_embeddings=pretrained_embeddings)
    else:
        raise ValueError(f"Unknown model name: {model_name}")