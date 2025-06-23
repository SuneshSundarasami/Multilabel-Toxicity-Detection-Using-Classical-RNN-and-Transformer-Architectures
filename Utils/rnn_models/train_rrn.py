import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import time
import urllib.request
import zipfile
import ssl
import multiprocessing

from rnn_models import (
    load_config, ToxicDataset, collate_batch, load_glove_embeddings,
     AdaptiveFocalLoss, create_vocabulary, tokens_to_indices, 
    oversample_minority_classes, get_model_by_name

)

def download_glove_300d(config):
    """Download GloVe 300d embeddings using config with SSL fix"""
    DATA_DIR = config['data']['glove_data_dir']
    os.makedirs(DATA_DIR, exist_ok=True)
    
    TXT_PATH = os.path.join(DATA_DIR, "glove.840B.300d.txt")
    
    if os.path.exists(TXT_PATH):
        print(f"GloVe 300d already exists at {TXT_PATH}")
        return TXT_PATH
    
    print("Downloading GloVe 300d (2.0GB)...")
    print("This may take 10-20 minutes depending on your internet speed...")
    
    # Create SSL context that doesn't verify certificates (for expired certs)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Alternative URLs to try
    urls = [
        "http://nlp.stanford.edu/data/glove.840B.300d.zip",
        "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    ]
    
    ZIP_PATH = os.path.join(DATA_DIR, "glove.840B.300d.zip")
    
    for i, url in enumerate(urls):
        try:
            print(f"Trying URL {i+1}: {url}")
            
            # Create a custom opener with the SSL context
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
            
            urllib.request.urlretrieve(url, ZIP_PATH, reporthook=progress_hook)
            print("\nDownload completed!")
            break
            
        except Exception as e:
            print(f"\nFailed with URL {i+1}: {e}")
            if i == len(urls) - 1:  # Last URL failed
                print("\nAll download attempts failed!")
                print("Please manually download GloVe 300d embeddings:")
                print("1. Go to: https://nlp.stanford.edu/projects/glove/")
                print("2. Download 'glove.840B.300d.zip' (Common Crawl 840B tokens)")
                print(f"3. Extract 'glove.840B.300d.txt' to: {DATA_DIR}")
                print("4. Re-run the script")
                return None
    
    # Extract the file
    print("Extracting...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Remove zip file to save space
        os.remove(ZIP_PATH)
        print(f"Done! Saved to {TXT_PATH}")
        return TXT_PATH
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    optimizer.zero_grad()
    
    for i, (tokens, labels) in enumerate(tqdm(dataloader, desc="Training")):
        tokens, labels = tokens.to(device), labels.to(device)
        
        outputs = model(tokens)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        
        total_loss += loss.item() * accumulation_steps
        total_accuracy += accuracy.item()
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Evaluating"):
            tokens, labels = tokens.to(device), labels.to(device)
            
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == labels).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def predict(model, dataloader, device):
    """Make predictions"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Predicting"):
            tokens, labels = tokens.to(device), labels.to(device)
            
            outputs = model(tokens)
            predictions = (outputs > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_predictions), np.vstack(all_labels)

def calculate_metrics(y_true, y_pred, label_names):
    """Calculate detailed metrics"""
    results = {}
    
    # Overall metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    results['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    
    # Per-class metrics
    for i, label in enumerate(label_names):
        results[f'f1_{label}'] = f1_score(y_true[:, i], y_pred[:, i])
    
    return results


def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=10, patience=3, save_path=None):
    """Complete training loop with F1-based early stopping"""
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Calculate FULL validation F1 (not sampled)
        val_predictions, val_true = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_predictions, average='macro', zero_division=0)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # F1-based early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"🎯 Model saved with F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{patience} (Best F1: {best_val_f1:.4f})")
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping after {epoch+1} epochs (Best F1: {best_val_f1:.4f})")
            break
    
    return train_losses, val_losses

def main():
    # Configuration
    CONFIG = load_config()
    
    # Hyperparameters - UPDATED FOR 300D
    EMBEDDING_DIM = 300  
    HIDDEN_DIM = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    DROPOUT = 0.3  
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Load preprocessed data - using config
    print("Loading preprocessed data...")
    with open(CONFIG['data']['train_tokens'], 'rb') as f:
        train_data = pickle.load(f)
    
    tokens = train_data['tokens']
    labels = train_data['labels']
    
    # Create vocabulary
    print("Creating vocabulary...")
    word2idx, idx2word = create_vocabulary(tokens, min_freq=2)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert tokens to indices
    print("Converting tokens to indices...")
    token_indices = tokens_to_indices(tokens, word2idx)
    
    # Split data
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        token_indices, labels, test_size=0.15, random_state=42, 
        stratify=labels[:, 0]  # Stratify by 'toxic' column
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42,
        stratify=y_temp[:, 0]
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Apply oversampling to training data
    print("Applying oversampling...")
    X_train_aug, y_train_aug = oversample_minority_classes(
        X_train, y_train, 
        multipliers={'threat': 3, 'identity_hate': 2, 'severe_toxic': 2}
    )
    print(f"After oversampling: {len(X_train_aug)}")
    
    # Create directories using config - CAPS VARIABLES
    RNN_ROOT = CONFIG['rnn_models']['root_dir']
    MODELS_DIR = os.path.join(RNN_ROOT, "models")
    RESULTS_DIR = os.path.join(RNN_ROOT, "results")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Download and load GloVe embeddings - CAPS VARIABLES
    GLOVE_PATH = download_glove_300d(CONFIG)
    
    if GLOVE_PATH and os.path.exists(GLOVE_PATH):
        print("Loading GloVe 300d embeddings...")
        EMBEDDING_MATRIX = load_glove_embeddings(GLOVE_PATH, word2idx, EMBEDDING_DIM)
    else:
        print("Using random embeddings instead of GloVe...")
        EMBEDDING_MATRIX = None
    
    # Get number of CPU cores
    NUM_WORKERS = multiprocessing.cpu_count()
    print(f"Using {NUM_WORKERS} CPU cores for data loading")
    
    # Create datasets and dataloaders - USING ALL CORES
    train_dataset = ToxicDataset(X_train_aug, y_train_aug)
    val_dataset = ToxicDataset(X_val, y_val)
    test_dataset = ToxicDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_batch, num_workers=NUM_WORKERS, 
                             pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           collate_fn=collate_batch, num_workers=NUM_WORKERS,
                           pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_batch, num_workers=NUM_WORKERS,
                            pin_memory=True, persistent_workers=True)
    
    # Calculate class weights for adaptive focal loss
    class_counts = np.sum(y_train_aug, axis=0)
    class_weights = len(y_train_aug) / (6 * class_counts)
    
    # Adjust class weights for rare classes (similar to your notebook)
    class_weights[3] *= 5   # threat
    class_weights[5] *= 3   # identity_hate  
    class_weights[1] *= 3   # severe_toxic
    
    print(f"Class weights: {class_weights}")
    print(f"Dropout rate: {DROPOUT}")  # Added logging for dropout
    
    # Define class-specific gamma values (from your notebook)
    class_gammas = {
        'toxic': 1.0,
        'severe_toxic': 1.5, 
        'obscene': 1.0,
        'threat': 1.5,       
        'insult': 1.0,
        'identity_hate': 2.0 
    }
    
    alpha = torch.FloatTensor(class_weights).to(DEVICE)
    
    # Model configurations
    models_config = {
        'GRU': 'gru',
        'LSTM': 'lstm', 
        'BiLSTM_Attention': 'bilstm_attention'
    }
    
    results = {}
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Train each model
    for model_name, model_type in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create model with custom dropout and n_layers=3
        if model_type == 'gru':
            from rnn_models import SimpleGRU
            model = SimpleGRU(
                vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 6,
                n_layers=3, dropout=DROPOUT, pretrained_embeddings=EMBEDDING_MATRIX
            ).to(DEVICE)
        elif model_type == 'lstm':
            from rnn_models import LSTMClassifier
            model = LSTMClassifier(
                vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 6,
                n_layers=3, dropout=DROPOUT, pretrained_embeddings=EMBEDDING_MATRIX
            ).to(DEVICE)
        elif model_type == 'bilstm_attention':
            from rnn_models import BiLSTMWithAttention
            model = BiLSTMWithAttention(
                vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 6,
                n_layers=3, dropout=DROPOUT, pretrained_embeddings=EMBEDDING_MATRIX
            ).to(DEVICE)

        criterion = AdaptiveFocalLoss(alpha=alpha, class_gammas=class_gammas).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train - using CAPS variables
        SAVE_PATH = os.path.join(MODELS_DIR, f"{model_name.lower()}_best.pt")
        
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, DEVICE,
            epochs=EPOCHS, patience=7, save_path=SAVE_PATH  # Increased epochs and patience
        )
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(SAVE_PATH))
        predictions, true_labels = predict(model, test_loader, DEVICE)
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, predictions, label_names)
        results[model_name] = metrics
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Micro F1: {metrics['micro_f1']:.4f}")
        
        # Per-class F1 scores
        print("\nPer-class F1 scores:")
        for label in label_names:
            print(f"{label}: {metrics[f'f1_{label}']:.4f}")
    
    # Save results - using CAPS variables
    results_df = pd.DataFrame(results).T
    RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "rnn_results.csv")
    results_df.to_csv(RESULTS_CSV_PATH)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(results_df[['accuracy', 'macro_f1', 'micro_f1']].round(4))
    
    # Save vocabulary - using CAPS variables
    VOCAB_PATH = os.path.join(MODELS_DIR, "vocabulary.pkl")
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    
    print("\nTraining completed! Models and results saved.")

if __name__ == "__main__":
    main()