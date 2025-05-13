from Utils.text_preprocessing import TextPreprocessor
from Utils.model_trainer import ModelTrainer
from Utils.model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import itertools
import time

# Load preprocessed data
train_data = pd.read_csv('./Dataset/train_preprocessed.csv')

# Fill NaN values
text_column = 'processed_text'
if text_column in train_data.columns:
    train_data[text_column] = train_data[text_column].fillna("")

# Split into train and test sets
train_data, test_data = train_test_split(
    train_data, 
    test_size=0.2, 
    random_state=42,
    stratify=train_data['toxic'] if 'toxic' in train_data.columns else None
)

print(f"Train set size: {train_data.shape[0]}")
print(f"Test set size: {test_data.shape[0]}")

processed_data = {
    'train': train_data,
    'test': test_data
}

# Create directories for results
model_dir = 'models'
results_dir = 'results'
for directory in [model_dir, results_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define all possible combinations to try
model_types = ['logistic', 'svm', 'rf', 'nb']
vectorizer_types = ['tfidf', 'count']
max_features_options = [5000, 10000, 20000]
ngram_ranges = [(1, 1), (1, 2), (1, 3)]

# Prepare for storing results
all_results = []
result_files = []
model_names = []

# Initialize trainer and evaluator
trainer = ModelTrainer(model_dir=model_dir)
evaluator = ModelEvaluator(results_dir=results_dir)

# Try all combinations
total_combinations = len(model_types) * len(vectorizer_types) * len(max_features_options) * len(ngram_ranges)
print(f"Testing {total_combinations} different model configurations...")

combination_count = 0
start_time = time.time()

for model_type, vectorizer_type, max_features, ngram_range in itertools.product(
    model_types, vectorizer_types, max_features_options, ngram_ranges):
    
    combination_count += 1
    config_name = f"{model_type}_{vectorizer_type}_f{max_features}_n{ngram_range[0]}-{ngram_range[1]}"
    
    print(f"\n[{combination_count}/{total_combinations}] Training configuration: {config_name}")
    
    # Prepare features with this configuration
    features = trainer.prepare_features(
        train_data=processed_data['train'],
        test_data=processed_data['test'],
        vectorizer_type=vectorizer_type,
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    # Train model
    try:
        model_results = trainer.train_model(
            data=features,
            model_type=model_type
        )
        
        # Evaluate model
        eval_results = evaluator.evaluate_model(
            model=model_results['model'],
            X_test=features['X_test'],
            y_test=features['y_test'],
            target_columns=features['target_columns'],
            output_name=config_name
        )
        
        # Store results for comparison
        result_files.append(f"results/{config_name}_results.pkl")
        model_names.append(config_name)
        
        # Store summary results
        all_results.append({
            'config': config_name,
            'model_type': model_type,
            'vectorizer_type': vectorizer_type,
            'max_features': max_features,
            'ngram_range': str(ngram_range),
            'accuracy': eval_results['accuracy'],
            'macro_f1': eval_results['macro_f1'],
            'micro_f1': eval_results['micro_f1'],
            'weighted_f1': eval_results['weighted_f1'],
            'mean_auc': eval_results.get('mean_auc', 0)
        })
        
    except Exception as e:
        print(f"Error training/evaluating config {config_name}: {str(e)}")
        continue
        
    # Show progress
    elapsed_time = time.time() - start_time
    avg_time_per_model = elapsed_time / combination_count
    remaining_models = total_combinations - combination_count
    est_remaining_time = avg_time_per_model * remaining_models
    
    print(f"Progress: {combination_count}/{total_combinations} configurations completed")
    print(f"Elapsed time: {elapsed_time:.1f} seconds")
    print(f"Estimated time remaining: {est_remaining_time:.1f} seconds ({est_remaining_time/60:.1f} minutes)")

# Compare all models
print("\nGenerating comparison of all model configurations...")
comparison = evaluator.compare_models(
    result_files=result_files,
    model_names=model_names,
    output_name='full_model_comparison'
)

# Create summary DataFrame and save it
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(results_dir, 'model_comparison_summary.csv'), index=False)

# Find and print the best performing models
print("\nTop 5 models by accuracy:")
print(summary_df.sort_values('accuracy', ascending=False).head(5)[['config', 'accuracy', 'macro_f1', 'mean_auc']])

print("\nTop 5 models by macro F1:")
print(summary_df.sort_values('macro_f1', ascending=False).head(5)[['config', 'accuracy', 'macro_f1', 'mean_auc']])

print("\nTop 5 models by AUC:")
print(summary_df.sort_values('mean_auc', ascending=False).head(5)[['config', 'accuracy', 'macro_f1', 'mean_auc']])

print(f"\nAll results saved to {os.path.join(results_dir, 'model_comparison_summary.csv')}")
print(f"Detailed comparison visualizations saved to {results_dir}")