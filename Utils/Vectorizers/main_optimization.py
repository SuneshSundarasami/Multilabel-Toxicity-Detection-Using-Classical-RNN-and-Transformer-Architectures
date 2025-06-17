import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import optuna
import yaml
import os
import psutil
import torch
from vectorizer_utils import get_vectorizers, get_models, rebuild_best_model, N_CORES, DEVICE

warnings.filterwarnings('ignore')

print(f"Using {N_CORES} CPU cores")
print(f"Using device: {DEVICE}")

# Load config
with open('../config/locations.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = config['data']['train_preprocessed']
RESULTS_PATH = '../results/optuna_vectorizer_allmodel_results.csv'

# Data Loading
train_data = pd.read_csv(DATA_PATH)
X = train_data['processed_text'].fillna("")
y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].fillna(0)

# Better data split approach
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['toxic']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp['toxic']
)

print(f"Train set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples") 
print(f"Test set: {len(X_test)} samples")

# Get vectorizers and models
vectorizers = get_vectorizers()
models = get_models()

def optimize_model(vec_name, vectorizer, model_name, objective_func):
    """Optimize a single model with given vectorizer"""
    try:
        print(f"  Optimizing {model_name} with {vec_name}...")
        
        # Create a fresh copy of the vectorizer for each run
        if hasattr(vectorizer, 'set_params'):
            vec_copy = vectorizer.__class__(**vectorizer.get_params())
        else:
            vec_copy = vectorizer
        
        X_train_vec = vec_copy.fit_transform(X_train)
        X_val_vec = vec_copy.transform(X_val)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective_func(trial, X_train_vec, y_train, X_val_vec, y_val),
            n_trials=20,
            n_jobs=1,
            show_progress_bar=False
        )
        
        best_score = study.best_value
        best_params = study.best_params
        
        result = {
            'vectorizer': vec_name,
            'model': model_name,
            'best_f1_score': best_score,
            'best_params': str(best_params)
        }
        
        print(f"    {vec_name} + {model_name}: Best F1 Score: {best_score:.4f}")
        return result
        
    except Exception as e:
        print(f"    Error optimizing {vec_name} + {model_name}: {str(e)}")
        return {
            'vectorizer': vec_name,
            'model': model_name,
            'best_f1_score': 0.0,
            'best_params': 'Error'
        }

def main():
    """Main optimization loop"""
    results = []
    
    for vec_name, vectorizer in vectorizers.items():
        print(f"\nRunning optimization for {vec_name} vectorizer...")
        
        # Run models sequentially for each vectorizer to avoid conflicts
        for model_name, objective_func in models.items():
            result = optimize_model(vec_name, vectorizer, model_name, objective_func)
            results.append(result)

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n=== Final Results ===")
    print(results_df.sort_values('best_f1_score', ascending=False))

    # Show top 5 best combinations
    print("\n=== Top 5 Best Performing Models ===")
    print(results_df.nlargest(5, 'best_f1_score')[['vectorizer', 'model', 'best_f1_score']])

    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    
    best_result = results_df.loc[results_df['best_f1_score'].idxmax()]
    best_vec_name = best_result['vectorizer']
    best_model_name = best_result['model']
    best_params_str = best_result['best_params']

    print(f"Best combination: {best_vec_name} + {best_model_name}")
    print(f"Validation F1 Score: {best_result['best_f1_score']:.4f}")

    # Train best model on train+val and evaluate on test
    best_vectorizer = vectorizers[best_vec_name]
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    X_trainval_vec = best_vectorizer.fit_transform(X_trainval)
    X_test_vec = best_vectorizer.transform(X_test)

    test_f1_score = rebuild_best_model(
        best_model_name, best_params_str, 
        X_trainval_vec, y_trainval, X_test_vec, y_test
    )
    
    print(f"Test F1 Score: {test_f1_score:.4f}")
    
    return results_df

if __name__ == "__main__":
    results_df = main()