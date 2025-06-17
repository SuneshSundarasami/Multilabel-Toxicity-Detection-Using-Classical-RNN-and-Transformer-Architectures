import pandas as pd
import numpy as np
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from vectorizer_utils import get_vectorizers, get_models, rebuild_best_model, N_CORES, DEVICE

# Paths
RESULTS_PATH = './results/optuna_vectorizer_allmodel_results.csv'
CONFIG_PATH = '../config/locations.yaml'

# Load config for data path
import yaml
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
DATA_PATH = config['data']['train_preprocessed']

# Load data
train_data = pd.read_csv(DATA_PATH)
X = train_data['processed_text'].fillna("")
y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].fillna(0)

# Split: train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['toxic']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp['toxic']
)
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

# Load vectorizers and models
vectorizers = get_vectorizers()
models = get_models()

# Load Optuna results
results_df = pd.read_csv(RESULTS_PATH)

# For storing validation results
validation_results = []

for idx, row in results_df.iterrows():
    vec_name = row['vectorizer']
    model_name = row['model']
    best_params_str = row['best_params']

    print(f"\nValidating {vec_name} + {model_name} on test set...")

    # Prepare vectorizer
    vectorizer = vectorizers[vec_name]
    X_trainval_vec = vectorizer.fit_transform(X_trainval)
    X_test_vec = vectorizer.transform(X_test)

    # Build model with best params
    test_macro_f1, test_micro_f1 = None, None
    try:
        # Use the rebuild_best_model utility if it returns both scores, else compute here
        model = rebuild_best_model(model_name, best_params_str, X_trainval_vec, y_trainval, X_test_vec, y_test)
        y_pred = model.predict(X_test_vec)
        test_macro_f1 = np.mean([
            f1_score(y_test.iloc[:, i], y_pred[:, i], average='macro')
            for i in range(y_test.shape[1])
        ])
        test_micro_f1 = np.mean([
            f1_score(y_test.iloc[:, i], y_pred[:, i], average='micro')
            for i in range(y_test.shape[1])
        ])
    except Exception as e:
        print(f"Error validating {vec_name} + {model_name}: {e}")
        test_macro_f1 = 0.0
        test_micro_f1 = 0.0

    print(f"Test Macro F1: {test_macro_f1:.4f} | Test Micro F1: {test_micro_f1:.4f}")

    validation_results.append({
        'vectorizer': vec_name,
        'model': model_name,
        'test_macro_f1': test_macro_f1,
        'test_micro_f1': test_micro_f1,
        'best_params': best_params_str
    })

# Save validation results
val_df = pd.DataFrame(validation_results)
val_df.to_csv('./results/final_test_f1_scores.csv', index=False)

print("\n=== Final Test F1 Scores ===")
print(val_df.sort_values('test_macro_f1', ascending=False))