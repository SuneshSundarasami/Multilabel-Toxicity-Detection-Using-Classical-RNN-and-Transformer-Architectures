import os
import json
import pandas as pd
from sklearn.metrics import f1_score
from vectorizer_utils import get_vectorizers, rebuild_best_model
import yaml

# Paths
INDIVIDUAL_RESULTS_DIR = '../results/individual_results/'
CONFIG_PATH = '../config/locations.yaml'

# Load config and test data
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
DATA_PATH = config['data']['train_preprocessed']
train_data = pd.read_csv(DATA_PATH)
X = train_data['processed_text'].fillna("")
y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].fillna(0)

# Use the same split as in main_optimization.py
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['toxic']
)

# Load vectorizers
vectorizers = get_vectorizers()

# Cache for fitted vectorizers
fitted_vectorizers = {}

results = []
print("Evaluating all models on test set...\n")
for idx, fname in enumerate(os.listdir(INDIVIDUAL_RESULTS_DIR)):
    if not fname.endswith('_result.json'):
        continue
    with open(os.path.join(INDIVIDUAL_RESULTS_DIR, fname), 'r') as f:
        result = json.load(f)
    if not result.get('completed', True):
        continue
    vec_name = result['vectorizer']
    model_name = result['model']
    best_params = result['best_params']

    print(f"[{idx+1}] Evaluating {vec_name} + {model_name}...")

    # Prepare vectorizer and data
    if vec_name in fitted_vectorizers:
        vectorizer, X_trainval_vec, X_test_vec = fitted_vectorizers[vec_name]
        print(f"  Using cached vectorizer: {vec_name}")
    else:
        vectorizer = vectorizers[vec_name]
        print(f"  Fitting vectorizer: {vec_name}")
        X_trainval_vec = vectorizer.fit_transform(X)  # Fit on all data
        X_test_vec = vectorizer.transform(X_test)
        fitted_vectorizers[vec_name] = (vectorizer, X_trainval_vec, X_test_vec)

    # Rebuild and predict
    print(f"  Rebuilding model: {model_name}")
    model = rebuild_best_model(model_name, str(best_params), X_trainval_vec, y, X_test_vec, y_test, return_model=True)
    print(f"  Model {model_name} rebuilt successfully.")
    print(f"  Predicting on test set...")
    y_pred = model.predict(X_test_vec)

    # Compute F1 scores
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print(f"  Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

    results.append({
        'vectorizer': vec_name,
        'model': model_name,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    })

print("\nSaving all results...")
df = pd.DataFrame(results)
df.to_csv('../results/all_models_test_f1_scores.csv', index=False)
print("Done! Top results:")
print(df.sort_values('macro_f1', ascending=False))