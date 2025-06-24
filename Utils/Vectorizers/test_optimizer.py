import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import os
import json

# Paths
SAVED_DATASETS_DIR = "/work/ssunda2s/toxic_comment_dataset/results/saved_datasets"
SAVED_MODELS_DIR = "/work/ssunda2s/toxic_comment_dataset/results/saved_models"
INDIVIDUAL_RESULTS_DIR = "/work/ssunda2s/toxic_comment_dataset/results/individual_results"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_RESULTS_DIR, exist_ok=True)

train_path = f"{SAVED_DATASETS_DIR}/train.csv"
val_path = f"{SAVED_DATASETS_DIR}/val.csv"
test_path = f"{SAVED_DATASETS_DIR}/test.csv"

# Load data
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

def get_X_y(df):
    X = df['processed_text'].fillna("").tolist()
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    return X, y

X_train, y_train = get_X_y(train_df)
X_val, y_val = get_X_y(val_df)
X_test, y_test = get_X_y(test_df)

def save_result(result, vec_name, model_name):
    filename = f"{vec_name}_{model_name}_result.json"
    filepath = os.path.join(INDIVIDUAL_RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {filename}")

# --- SentenceTransformer + LightGBM (with class_weight) ---
sent_trans_name = "sentence-transformers/all-MiniLM-L6-v2"
sent_model = SentenceTransformer(sent_trans_name)

def embed_sentences(model, texts, batch_size=128):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

print("\n[SentenceTransformer + LightGBM]")
X_train_emb = embed_sentences(sent_model, X_train)
X_val_emb = embed_sentences(sent_model, X_val)
X_test_emb = embed_sentences(sent_model, X_test)

# LightGBM with class_weight='balanced'
lgbm_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 8, 'class_weight': 'balanced'}
model = MultiOutputClassifier(lgb.LGBMClassifier(**lgbm_params), n_jobs=8)
model.fit(np.vstack([X_train_emb, X_val_emb]), np.vstack([y_train, y_val]))
y_pred = model.predict(X_test_emb)

macro_f1 = f1_score(y_test, y_pred, average="macro")
class_f1 = f1_score(y_test, y_pred, average=None)
result = {
    "vectorizer": "SentenceTransformer-MiniLM-L6-v2",
    "model": "LightGBM_classweight",
    "macro_f1": macro_f1,
    "class_f1": dict(zip(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], class_f1))
}
print("Macro F1:", macro_f1)
print("Class-wise F1:", result["class_f1"])
print(classification_report(y_test, y_pred, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))

# Save model and embeddings vectorizer
vec_path = os.path.join(SAVED_MODELS_DIR, "SentenceTransformer-MiniLM-L6-v2_vectorizer.joblib")
model_path = os.path.join(SAVED_MODELS_DIR, "SentenceTransformer-MiniLM-L6-v2_LightGBM_classweight_model.joblib")
joblib.dump(sent_model, vec_path)
joblib.dump(model, model_path)
save_result(result, "SentenceTransformer-MiniLM-L6-v2", "LightGBM_classweight")

# --- BERT + LightGBM (with class_weight) ---
bert_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name)
bert_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = bert_model.to(device)

def bert_embed(texts, tokenizer, model, device, batch_size=32, max_length=128):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(emb)
    return np.vstack(all_embeddings)

print("\n[BERT + LightGBM]")
X_train_emb = bert_embed(X_train, tokenizer, bert_model, device)
X_val_emb = bert_embed(X_val, tokenizer, bert_model, device)
X_test_emb = bert_embed(X_test, tokenizer, bert_model, device)

model = MultiOutputClassifier(lgb.LGBMClassifier(**lgbm_params), n_jobs=8)
model.fit(np.vstack([X_train_emb, X_val_emb]), np.vstack([y_train, y_val]))
y_pred = model.predict(X_test_emb)

macro_f1 = f1_score(y_test, y_pred, average="macro")
class_f1 = f1_score(y_test, y_pred, average=None)
result = {
    "vectorizer": "BERT-base-uncased",
    "model": "LightGBM_classweight",
    "macro_f1": macro_f1,
    "class_f1": dict(zip(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], class_f1))
}
print("Macro F1:", macro_f1)
print("Class-wise F1:", result["class_f1"])
print(classification_report(y_test, y_pred, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))

vec_path = os.path.join(SAVED_MODELS_DIR, "BERT-base-uncased_vectorizer.joblib")
model_path = os.path.join(SAVED_MODELS_DIR, "BERT-base-uncased_LightGBM_classweight_model.joblib")
joblib.dump((tokenizer, bert_model), vec_path)
joblib.dump(model, model_path)
save_result(result, "BERT-base-uncased", "LightGBM_classweight")