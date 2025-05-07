
from Utils.text_preprocessing import TextPreprocessor
from Utils.text_preprocessing import TextPreprocessor
from Utils.model_trainer import ModelTrainer
from Utils.model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split



import pandas as pd

# Initialize the preprocessor
preprocessor = TextPreprocessor(download_nltk_resources=True, keep_negations=True)

# # Preprocess datasets
# processed_data = preprocessor.preprocess_dataset(
#     train_path='./Dataset/train.csv',
#     test_path='./Dataset/test.csv',
#     test_labels_path='./Dataset/test_labels.csv',
#     output_dir='./Dataset',
#     remove_stops=True,
#     lemmatize=True,
#     stem=False
# )

# Load preprocessed data instead of preprocessing again
train_data = pd.read_csv('./Dataset/train_preprocessed.csv')
# test_data = pd.read_csv('./Dataset/test_preprocessed.csv')

train_data, test_data = train_test_split(
    train_data, 
    test_size=0.2, 
    random_state=42,
    stratify=train_data['toxic'] if 'toxic' in train_data.columns else None
)

processed_data = {
    'train': train_data,
    'test': test_data
}

# Train model
trainer = ModelTrainer(model_dir='models')
features = trainer.prepare_features(
    train_data=processed_data['train'],
    test_data=processed_data['test'],
    vectorizer_type='tfidf'
)

model_results = trainer.train_model(
    data=features,
    model_type='logistic'
)

# Evaluate model
evaluator = ModelEvaluator(results_dir='results')
eval_results = evaluator.evaluate_model(
    model=model_results['model'],
    X_test=features['X_test'],
    y_test=features['y_test'],
    target_columns=features['target_columns'],
    output_name='logistic_tfidf'
)

# Find optimal thresholds
thresholds = evaluator.find_optimal_threshold(
    model=model_results['model'],
    X_val=features['X_test'],  # Using test data as validation for example
    y_val=features['y_test'],
    target_columns=features['target_columns'],
    output_name='logistic_tfidf'
)

# Train multiple models and compare them
models = ['logistic', 'svm', 'rf']
result_files = []

for model_type in models:
    model_results = trainer.train_model(
        data=features,
        model_type=model_type
    )
    
    output_name = f"{model_type}_tfidf"
    eval_results = evaluator.evaluate_model(
        model=model_results['model'],
        X_test=features['X_test'],
        y_test=features['y_test'],
        target_columns=features['target_columns'],
        output_name=output_name
    )
    
    result_files.append(f"results/{output_name}_results.pkl")

# Compare models
comparison = evaluator.compare_models(
    result_files=result_files,
    model_names=models,
    output_name='model_comparison'
)