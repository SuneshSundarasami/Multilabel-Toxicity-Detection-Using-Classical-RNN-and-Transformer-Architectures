import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
import json
import glob
import joblib

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')

import optuna
import yaml
import os
import psutil
import torch
from vectorizer_utils import get_vectorizers, get_models, rebuild_best_model, N_CORES, DEVICE

print(f"Using {N_CORES} CPU cores")
print(f"Using device: {DEVICE}")

# Load config
with open('../config/locations.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get all paths from config
DATA_PATH = config['data']['train_preprocessed']
RESULTS_PATH = config['results']['optuna_vectorizer_allmodel_results']
INDIVIDUAL_RESULTS_DIR = config['results']['individual_results']
SAVED_MODELS_DIR = config['results']['saved_models']
SAVED_DATASETS_DIR = config['results']['saved_datasets']

# Create results directories
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(INDIVIDUAL_RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_DATASETS_DIR, exist_ok=True)

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

# Save datasets (once, outside the loop)
pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(SAVED_DATASETS_DIR, "train.csv"), index=False)
pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(SAVED_DATASETS_DIR, "val.csv"), index=False)
pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(SAVED_DATASETS_DIR, "test.csv"), index=False)

# Get vectorizers and models
vectorizers = get_vectorizers()
models = get_models()

def save_individual_result(result, vec_name, model_name):
    """Save individual result to a separate file"""
    filename = f"{vec_name}_{model_name}_result.json"
    filepath = os.path.join(INDIVIDUAL_RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"    Saved result to {filename}")

def load_existing_results():
    """Load all existing individual results"""
    results = []
    pattern = os.path.join(INDIVIDUAL_RESULTS_DIR, "*_result.json")
    
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results

def check_if_completed(vec_name, model_name):
    """Check if this vectorizer-model combination is already completed"""
    filename = f"{vec_name}_{model_name}_result.json"
    filepath = os.path.join(INDIVIDUAL_RESULTS_DIR, filename)
    return os.path.exists(filepath)

def optimize_model(vec_name, vectorizer, model_name, objective_func):
    """Optimize a single model with given vectorizer"""
    # Check if already completed
    if check_if_completed(vec_name, model_name):
        print(f"  Skipping {model_name} with {vec_name} (already completed)")
        return None
    
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
            n_trials=10,  # Reduced for faster testing
            n_jobs=1,
            show_progress_bar=False
        )
        
        best_score = study.best_value
        best_params = study.best_params
        
        result = {
            'vectorizer': vec_name,
            'model': model_name,
            'best_f1_score': best_score,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'completed': True
        }
        
        # Save individual result immediately
        save_individual_result(result, vec_name, model_name)
        
        print(f"    {vec_name} + {model_name}: Best F1 Score: {best_score:.4f}")

        # --- Save fitted vectorizer and model for this combo ---
        try:
            vec_filename = f"{vec_name}_vectorizer.joblib"
            model_filename = f"{vec_name}_{model_name}_model.joblib"
            vec_path = os.path.join(SAVED_MODELS_DIR, vec_filename)
            model_path = os.path.join(SAVED_MODELS_DIR, model_filename)

            # Fit vectorizer on train+val
            X_trainval = pd.concat([X_train, X_val])
            y_trainval = pd.concat([y_train, y_val])
            fitted_vectorizer = vectorizer.__class__(**vectorizer.get_params())
            X_trainval_vec = fitted_vectorizer.fit_transform(X_trainval)

            # Rebuild and fit the best model on train+val
            best_model = rebuild_best_model(model_name, str(best_params), X_trainval_vec, y_trainval, X_trainval_vec, y_trainval, return_model=True)

            joblib.dump(fitted_vectorizer, vec_path)
            joblib.dump(best_model, model_path)
            print(f"    Saved vectorizer to {vec_path}")
            print(f"    Saved model to {model_path}")
        except Exception as e:
            print(f"    Could not save model/vectorizer for {vec_name} + {model_name}: {e}")

        return result
        
    except Exception as e:
        error_result = {
            'vectorizer': vec_name,
            'model': model_name,
            'best_f1_score': 0.0,
            'best_params': {},
            'error': str(e),
            'completed': False
        }
        
        # Save error result too
        save_individual_result(error_result, vec_name, model_name)
        
        print(f"    Error optimizing {vec_name} + {model_name}: {str(e)}")
        return error_result

def combine_all_results():
    """Combine all individual results into final CSV"""
    all_results = load_existing_results()
    
    if not all_results:
        print("No results found to combine!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Convert best_params back to string for CSV compatibility
    results_df['best_params_str'] = results_df['best_params'].astype(str)
    
    # Save combined results
    results_df.to_csv(RESULTS_PATH, index=False)
    
    print(f"\nCombined {len(all_results)} results and saved to {RESULTS_PATH}")
    return results_df

def main():
    """Main optimization loop"""
    print("Loading existing results...")
    existing_results = load_existing_results()
    print(f"Found {len(existing_results)} existing results")
    
    # Get all vectorizer-model combinations
    total_combinations = len(vectorizers) * len(models)
    completed_combinations = len(existing_results)
    
    print(f"Progress: {completed_combinations}/{total_combinations} combinations completed")
    
    # Run optimization for missing combinations
    for vec_name, vectorizer in vectorizers.items():
        print(f"\nProcessing {vec_name} vectorizer...")
        
        for model_name, objective_func in models.items():
            optimize_model(vec_name, vectorizer, model_name, objective_func)
    
    # Combine all results
    results_df = combine_all_results()
    
    if len(results_df) > 0:
        print("\n=== Final Results ===")
        completed_df = results_df[results_df.get('completed', True) == True]
        if len(completed_df) > 0:
            print(completed_df.sort_values('best_f1_score', ascending=False)[['vectorizer', 'model', 'best_f1_score']])

            # Show top 5 best combinations
            print("\n=== Top 5 Best Performing Models ===")
            top5 = completed_df.nlargest(5, 'best_f1_score')[['vectorizer', 'model', 'best_f1_score']]
            print(top5)

            # Final evaluation on test set with best model
            print("\n=== Final Evaluation on Test Set ===")
            
            best_result = completed_df.loc[completed_df['best_f1_score'].idxmax()]
            best_vec_name = best_result['vectorizer']
            best_model_name = best_result['model']
            best_params = best_result['best_params']

            print(f"Best combination: {best_vec_name} + {best_model_name}")
            print(f"Validation F1 Score: {best_result['best_f1_score']:.4f}")

            # Train best model on train+val and evaluate on test
            best_vectorizer = vectorizers[best_vec_name]
            X_trainval = pd.concat([X_train, X_val])
            y_trainval = pd.concat([y_train, y_val])

            X_trainval_vec = best_vectorizer.fit_transform(X_trainval)
            X_test_vec = best_vectorizer.transform(X_test)

            test_f1_score = rebuild_best_model(
                best_model_name, str(best_params), 
                X_trainval_vec, y_trainval, X_test_vec, y_test
            )
            
            print(f"Test F1 Score: {test_f1_score:.4f}")
            
            # Save the fitted vectorizer and model
            vec_filename = f"{vec_name}_vectorizer.joblib"
            model_filename = f"{vec_name}_{model_name}_model.joblib"
            vec_path = os.path.join(SAVED_MODELS_DIR, vec_filename)
            model_path = os.path.join(SAVED_MODELS_DIR, model_filename)

            # Fit on train+val for saving
            X_trainval = pd.concat([X_train, X_val])
            y_trainval = pd.concat([y_train, y_val])
            fitted_vectorizer = vectorizer.__class__(**vectorizer.get_params())
            X_trainval_vec = fitted_vectorizer.fit_transform(X_trainval)
            best_model = rebuild_best_model(model_name, str(best_params), X_trainval_vec, y_trainval, X_trainval_vec, y_trainval, return_model=True)

            joblib.dump(fitted_vectorizer, vec_path)
            joblib.dump(best_model, model_path)
            print(f"    Saved vectorizer to {vec_path}")
            print(f"    Saved model to {model_path}")
    
    return results_df

if __name__ == "__main__":
    results_df = main()