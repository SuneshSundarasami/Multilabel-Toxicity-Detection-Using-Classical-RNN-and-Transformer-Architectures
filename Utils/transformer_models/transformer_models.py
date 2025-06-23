import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'locations.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

class TransformerToxicClassifier(nn.Module):
    def __init__(self, model_name, num_labels=6, dropout=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return torch.sigmoid(logits)

class TransformerToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.FloatTensor(self.labels[idx])
        return item

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return input_ids, attention_mask, labels

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0, 0
    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = (outputs > 0.5).float()
        acc = (preds == labels).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(dataloader), total_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            preds = (outputs > 0.5).float()
            acc = (preds == labels).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss / len(dataloader), total_acc / len(dataloader)

def predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Predicting"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return np.vstack(all_preds), np.vstack(all_labels)

def calculate_metrics(y_true, y_pred, label_names):
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    results['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    for i, label in enumerate(label_names):
        results[f'f1_{label}'] = f1_score(y_true[:, i], y_pred[:, i])
    return results

def oversample_minority_classes(X, y, multipliers=None):
    """
    Oversample minority classes in a multi-label setting.
    multipliers: dict mapping class index or name to oversampling multiplier.
    """
    X_aug, y_aug = list(X), list(y)
    y = np.array(y)
    if multipliers is None:
        return X_aug, y_aug
    class_indices = {
        'toxic': 0, 'severe_toxic': 1, 'obscene': 2,
        'threat': 3, 'insult': 4, 'identity_hate': 5
    }
    for cls, mult in multipliers.items():
        idx = class_indices[cls] if isinstance(cls, str) else cls
        pos_indices = np.where(y[:, idx] == 1)[0]
        for _ in range(mult - 1):
            for i in pos_indices:
                X_aug.append(X[i])
                y_aug.append(y[i])
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    return X_aug, y_aug

class AdaptiveFocalLoss(nn.Module):
    """
    Multi-label focal loss with per-class alpha and gamma.
    """
    def __init__(self, alpha=None, class_gammas=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.class_gammas = class_gammas
        self.reduction = reduction
        self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def forward(self, inputs, targets):
        # inputs: (batch, num_classes), targets: (batch, num_classes)
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        gamma = torch.ones(inputs.shape[1], device=inputs.device)
        if self.class_gammas:
            for i, name in enumerate(self.class_names):
                gamma[i] = self.class_gammas.get(name, 1.0)
        loss = (self.alpha * (1 - pt) ** gamma) * BCE_loss if self.alpha is not None else ((1 - pt) ** gamma) * BCE_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss