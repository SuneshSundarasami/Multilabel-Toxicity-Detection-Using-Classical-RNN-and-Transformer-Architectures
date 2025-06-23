import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.metrics import f1_score, accuracy_score, classification_report


from transformer_models import (
    TransformerToxicClassifier, get_tokenizer, TransformerToxicDataset,
    collate_fn, train_epoch, evaluate, predict, calculate_metrics, load_config, oversample_minority_classes, AdaptiveFocalLoss
)

def train_and_evaluate(model_name, model_tag,batch_size=100, epochs=5, learning_rate=2e-5):
    CONFIG = load_config()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 128
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = learning_rate

    print(f"\n=== Training {model_tag} ({model_name}) ===")
    print(f"Using device: {DEVICE}")

    # Load data
    with open(CONFIG['data']['train_tokens'], 'rb') as f:
        train_data = pickle.load(f)
    texts = [" ".join(tokens) for tokens in train_data['tokens']]
    labels = train_data['labels']

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels[:, 0]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp[:, 0]
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Oversample
    print("Applying oversampling...")
    X_train_aug, y_train_aug = oversample_minority_classes(
        X_train, y_train, multipliers={'threat': 3, 'identity_hate': 2, 'severe_toxic': 2}
    )
    print(f"After oversampling: {len(X_train_aug)}")

    # Prepare tokenizer and datasets
    tokenizer = get_tokenizer(model_name)
    train_dataset = TransformerToxicDataset(X_train_aug, y_train_aug, tokenizer, MAX_LEN)
    val_dataset = TransformerToxicDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = TransformerToxicDataset(X_test, y_test, tokenizer, MAX_LEN)

    NUM_WORKERS = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Class weights and gamma (same as RNN)
    class_counts = np.sum(y_train_aug, axis=0)
    class_weights = len(y_train_aug) / (6 * class_counts)
    class_weights[3] *= 5
    class_weights[5] *= 3
    class_weights[1] *= 3
    class_gammas = {
        'toxic': 1.0, 'severe_toxic': 1.5, 'obscene': 1.0,
        'threat': 1.5, 'insult': 1.0, 'identity_hate': 2.0
    }
    alpha = torch.FloatTensor(class_weights).to(DEVICE)

    # Model, loss, optimizer
    model = TransformerToxicClassifier(model_name, num_labels=6, dropout=0.3).to(DEVICE)
    criterion = AdaptiveFocalLoss(alpha=alpha, class_gammas=class_gammas).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_f1 = 0
    patience, patience_counter = 3, 0
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    TRANSFORMER_ROOT = CONFIG['transformer_models']['root_dir']
    os.makedirs(TRANSFORMER_ROOT, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(TRANSFORMER_ROOT, f"{model_tag}_best.pt")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        val_preds, val_true = predict(model, val_loader, DEVICE)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🎯 Model saved with F1: {val_f1:.4f} at {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{patience} (Best F1: {best_val_f1:.4f})")
        if patience_counter >= patience:
            print(f"🛑 Early stopping after {epoch+1} epochs (Best F1: {best_val_f1:.4f})")
            break

    # Evaluate on test set
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_preds, test_true = predict(model, test_loader, DEVICE)
    metrics = calculate_metrics(test_true, test_preds, label_names)
    print(f"\nTest Results for {model_tag}:")
    print(metrics)
    print("\nClassification Report (Test Set):")
    print(classification_report(test_true, test_preds, target_names=label_names, zero_division=0))

    # Save results to CSV
    RESULTS_DIR = os.path.join(TRANSFORMER_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{model_tag}_test_metrics.csv")
    import csv
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f"Test metrics saved to {results_path}")

def main():
    # Train with plain BERT
    train_and_evaluate("bert-base-uncased", "bert_base", batch_size=16, epochs=8, learning_rate=5e-5)
    # Train with DeBERTa-v3-large (best as of 2025)
    train_and_evaluate("microsoft/deberta-v3-large", "deberta_v3_large", batch_size=32, epochs=3, learning_rate=5e-5)

if __name__ == "__main__":
    main()