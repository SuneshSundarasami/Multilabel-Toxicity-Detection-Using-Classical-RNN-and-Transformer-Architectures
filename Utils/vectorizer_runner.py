#!/usr/bin/env python3
# vectorizer_runner.py

import argparse
import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
import multiprocessing
import warnings
import torch
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)

# Set constants
RESULTS_DIR = '../results/vectorizers_models'
MODELS_DIR = '../models/vectorizers'
TEMP_DIR = '../temp_vectors'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Check for GPU
gpu_available = torch.cuda.is_available()
if gpu_available:
    print(f"GPU acceleration available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU only")

# Get available CPU cores
n_cores = multiprocessing.cpu_count()
print(f"Available CPU cores: {n_cores}")


# Define Vectorizers
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg  # 0 for CBOW, 1 for Skip-gram
        self.model = None
        self.word_vectors = None
        
    def fit(self, X, y=None):
        # Tokenize the text
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in tqdm(X, desc="Tokenizing for Word2Vec")]
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        
        self.word_vectors = self.model.wv
        print(f"Word2Vec model trained with {len(self.word_vectors.key_to_index)} words")
        return self
    
    def transform(self, X):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in tqdm(X, desc="Vectorizing with Word2Vec")]
        
        # Create document vectors by averaging word vectors
        doc_vectors = np.zeros((len(tokenized_corpus), self.vector_size))
        for i, tokens in enumerate(tokenized_corpus):
            vec = np.zeros(self.vector_size)
            count = 0
            for token in tokens:
                if token in self.word_vectors:
                    vec += self.word_vectors[token]
                    count += 1
            if count > 0:
                vec /= count
            doc_vectors[i] = vec
        
        return doc_vectors


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        
    def fit(self, X, y=None):
        # Tokenize the text
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in tqdm(X, desc="Tokenizing for FastText")]
        
        # Train FastText model
        print("Training FastText model...")
        self.model = FastText(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        
        print(f"FastText model trained with {len(self.model.wv.key_to_index)} words")
        return self
    
    def transform(self, X):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in tqdm(X, desc="Vectorizing with FastText")]
        
        # Create document vectors by averaging word vectors
        doc_vectors = np.zeros((len(tokenized_corpus), self.vector_size))
        for i, tokens in enumerate(tokenized_corpus):
            vec = np.zeros(self.vector_size)
            count = 0
            for token in tokens:
                # FastText can handle OOV words
                vec += self.model.wv[token]
                count += 1
            if count > 0:
                vec /= count
            doc_vectors[i] = vec
        
        return doc_vectors


class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.word_vectors = {}
        
    def fit(self, X, y=None):
        # Attempt to download pre-trained GloVe using gensim downloader
        try:
            import gensim.downloader as api
            print("Downloading pre-trained GloVe embeddings...")
            # Use a smaller model for demonstration
            glove_model = api.load("glove-wiki-gigaword-100")
            self.word_vectors = {word: glove_model[word] for word in glove_model.key_to_index}
            print(f"Loaded GloVe embeddings with {len(self.word_vectors)} words")
        except Exception as e:
            print(f"Error loading GloVe: {e}")
            print("Will use an empty embedding. Results will be poor.")
        
        return self
    
    def transform(self, X):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in tqdm(X, desc="Vectorizing with GloVe")]
        
        # Create document vectors by averaging word vectors
        doc_vectors = np.zeros((len(tokenized_corpus), self.vector_size))
        for i, tokens in enumerate(tokenized_corpus):
            vec = np.zeros(self.vector_size)
            count = 0
            for token in tokens:
                if token in self.word_vectors:
                    vec += self.word_vectors[token]
                    count += 1
            if count > 0:
                vec /= count
            doc_vectors[i] = vec
        
        return doc_vectors


# Evaluation functions
def evaluate_model(model, X, y, model_name):
    start_time = time.time()
    y_pred = model.predict(X)
    inference_time = time.time() - start_time
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y, y_pred)
    
    # Calculate F1 scores for each class
    f1_scores = []
    for i, column in enumerate(y.columns):
        f1 = f1_score(y[column], y_pred[:, i])
        f1_scores.append(f1)
    
    # Calculate macro and micro F1
    macro_f1 = np.mean(f1_scores)
    micro_f1 = f1_score(y, y_pred, average='micro')
    
    print(f"\n============ {model_name} Results ============")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    
    print("\nF1 scores by toxicity type:")
    for i, column in enumerate(y.columns):
        f1 = f1_score(y[column], y_pred[:, i])
        print(f"{column}: {f1:.4f}")
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'inference_time': inference_time
    }
    
    for i, column in enumerate(y.columns):
        results[f'f1_{column}'] = f1_scores[i]
    
    return results, y_pred


def evaluate_vectorizer_with_classifiers(X_train, X_val, y_train, y_val, vectorizer_name, n_cores):
    """
    Evaluate multiple classifiers on vectorized features
    """
    # Define classifiers to evaluate
    classifiers = {
        'SVM': MultiOutputClassifier(LinearSVC(
            C=1.0, 
            max_iter=10000, 
            dual=False, 
            class_weight='balanced',
            random_state=42
        )),
        
        'LogisticRegression': MultiOutputClassifier(LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            n_jobs=n_cores
        )),
        
        'KNeighborsClassifier': MultiOutputClassifier(KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=n_cores
        )),
    }
    
    # For XGBoost, configure GPU if available
    if gpu_available:
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': n_cores,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
    else:
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': n_cores
        }
        
    classifiers['XGBoost'] = MultiOutputClassifier(xgb.XGBClassifier(**xgb_params))
    
    results = []
    
    # Evaluate each classifier
    for clf_name, classifier in classifiers.items():
        model_name = f"{vectorizer_name} + {clf_name}"
        print(f"\n===== Training {model_name} =====")
        
        # Train classifier
        start_time = time.time()
        classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Create a wrapper for evaluation
        class SimpleModelWrapper:
            def __init__(self, clf):
                self.clf = clf
            def predict(self, X):
                return self.clf.predict(X)
        
        model = SimpleModelWrapper(classifier)
        result, preds = evaluate_model(model, X_val, y_val, model_name)
        result['train_time'] = train_time
        results.append(result)
        
    return results


def clean_up_temp_files():
    """Remove temporary files to free up memory"""
    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Force garbage collection
    gc.collect()


def process_tfidf_vectorizer(X_train, X_val, y_train, y_val):
    """Process TF-IDF vectorization and evaluation"""
    print("\n\n================ PROCESSING TF-IDF VECTORIZER ================")
    
    # Initialize and fit vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    print("Fitting TF-IDF vectorizer...")
    start_time = time.time()
    tfidf_vectorizer.fit(X_train)
    train_time = time.time() - start_time
    print(f"TF-IDF vectorizer fitted in {train_time:.2f} seconds")
    
    # Transform data
    print("Transforming data with TF-IDF...")
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    print(f"TF-IDF vectors shape: {X_train_tfidf.shape}")
    
    # Save vectorizer
    with open(f'{MODELS_DIR}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF vectorizer saved to {MODELS_DIR}/tfidf_vectorizer.pkl")
    
    # Evaluate with classifiers
    results = evaluate_vectorizer_with_classifiers(
        X_train_tfidf, X_val_tfidf, y_train, y_val, "TF-IDF", n_cores
    )
    
    # Free memory
    del X_train_tfidf, X_val_tfidf, tfidf_vectorizer
    gc.collect()
    
    return results, {"TF-IDF": X_train_tfidf.shape}


def process_count_vectorizer(X_train, X_val, y_train, y_val):
    """Process Count vectorization and evaluation"""
    print("\n\n================ PROCESSING COUNT VECTORIZER ================")
    
    # Initialize and fit vectorizer
    count_vectorizer = CountVectorizer(
        max_features=20000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    print("Fitting Count vectorizer...")
    start_time = time.time()
    count_vectorizer.fit(X_train)
    train_time = time.time() - start_time
    print(f"Count vectorizer fitted in {train_time:.2f} seconds")
    
    # Transform data
    print("Transforming data with Count Vectorizer...")
    X_train_count = count_vectorizer.transform(X_train)
    X_val_count = count_vectorizer.transform(X_val)
    print(f"Count vectors shape: {X_train_count.shape}")
    
    # Save vectorizer
    with open(f'{MODELS_DIR}/count_vectorizer.pkl', 'wb') as f:
        pickle.dump(count_vectorizer, f)
    print(f"Count vectorizer saved to {MODELS_DIR}/count_vectorizer.pkl")
    
    # Define one classifier - LogisticRegression for Count Vectorizer
    classifier = MultiOutputClassifier(LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        n_jobs=n_cores
    ))
    
    model_name = "Count + LogisticRegression"
    print(f"\n===== Training {model_name} =====")
    
    # Train classifier
    start_time = time.time()
    classifier.fit(X_train_count, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Create a wrapper for evaluation
    class SimpleModelWrapper:
        def __init__(self, clf):
            self.clf = clf
        def predict(self, X):
            return self.clf.predict(X)
    
    model = SimpleModelWrapper(classifier)
    result, preds = evaluate_model(model, X_val_count, y_val, model_name)
    result['train_time'] = train_time
    
    # Free memory
    del X_train_count, X_val_count, count_vectorizer
    gc.collect()
    
    return [result], {"Count": X_train_count.shape}


def process_word2vec_vectorizer(X_train, X_val, y_train, y_val):
    """Process Word2Vec vectorization and evaluation"""
    print("\n\n================ PROCESSING WORD2VEC VECTORIZER ================")
    
    # Initialize and fit vectorizer
    w2v_vectorizer = Word2VecVectorizer(
        vector_size=500, 
        window=5, 
        min_count=1, 
        workers=n_cores, 
        sg=1
    )
    
    print("Fitting Word2Vec vectorizer...")
    start_time = time.time()
    w2v_vectorizer.fit(X_train)
    train_time = time.time() - start_time
    print(f"Word2Vec training completed in {train_time:.2f} seconds")
    
    # Transform data
    print("Transforming training data with Word2Vec...")
    X_train_w2v = w2v_vectorizer.transform(X_train)
    print("Transforming validation data with Word2Vec...")
    X_val_w2v = w2v_vectorizer.transform(X_val)
    print(f"Word2Vec vectors shape: {X_train_w2v.shape}")
    
    # Save vectorizer
    with open(f'{MODELS_DIR}/word2vec_vectorizer.pkl', 'wb') as f:
        pickle.dump(w2v_vectorizer, f)
    print(f"Word2Vec vectorizer saved to {MODELS_DIR}/word2vec_vectorizer.pkl")
    
    # Evaluate with classifiers
    results = evaluate_vectorizer_with_classifiers(
        X_train_w2v, X_val_w2v, y_train, y_val, "Word2Vec", n_cores
    )
    
    # Free memory
    del X_train_w2v, X_val_w2v, w2v_vectorizer
    gc.collect()
    
    return results, {"Word2Vec": X_train_w2v.shape}


def process_fasttext_vectorizer(X_train, X_val, y_train, y_val):
    """Process FastText vectorization and evaluation"""
    print("\n\n================ PROCESSING FASTTEXT VECTORIZER ================")
    
    # Initialize and fit vectorizer
    fasttext_vectorizer = FastTextVectorizer(
        vector_size=500, 
        window=5, 
        min_count=1, 
        workers=n_cores
    )
    
    print("Fitting FastText vectorizer...")
    start_time = time.time()
    fasttext_vectorizer.fit(X_train)
    train_time = time.time() - start_time
    print(f"FastText training completed in {train_time:.2f} seconds")
    
    # Transform data
    print("Transforming training data with FastText...")
    X_train_fasttext = fasttext_vectorizer.transform(X_train)
    print("Transforming validation data with FastText...")
    X_val_fasttext = fasttext_vectorizer.transform(X_val)
    print(f"FastText vectors shape: {X_train_fasttext.shape}")
    
    # Save vectorizer
    with open(f'{MODELS_DIR}/fasttext_vectorizer.pkl', 'wb') as f:
        pickle.dump(fasttext_vectorizer, f)
    print(f"FastText vectorizer saved to {MODELS_DIR}/fasttext_vectorizer.pkl")
    
    # Evaluate with classifiers
    results = evaluate_vectorizer_with_classifiers(
        X_train_fasttext, X_val_fasttext, y_train, y_val, "FastText", n_cores
    )
    
    # Free memory
    del X_train_fasttext, X_val_fasttext, fasttext_vectorizer
    gc.collect()
    
    return results, {"FastText": X_train_fasttext.shape}


def process_glove_vectorizer(X_train, X_val, y_train, y_val):
    """Process GloVe vectorization and evaluation"""
    print("\n\n================ PROCESSING GLOVE VECTORIZER ================")
    
    # Initialize and fit vectorizer
    glove_vectorizer = GloveVectorizer(vector_size=500)
    
    print("Fitting GloVe vectorizer...")
    start_time = time.time()
    glove_vectorizer.fit(X_train)
    train_time = time.time() - start_time
    print(f"GloVe preparation completed in {train_time:.2f} seconds")
    
    # Transform data
    print("Transforming training data with GloVe...")
    X_train_glove = glove_vectorizer.transform(X_train)
    print("Transforming validation data with GloVe...")
    X_val_glove = glove_vectorizer.transform(X_val)
    print(f"GloVe vectors shape: {X_train_glove.shape}")
    
    # Save vectorizer
    with open(f'{MODELS_DIR}/glove_vectorizer.pkl', 'wb') as f:
        pickle.dump(glove_vectorizer, f)
    print(f"GloVe vectorizer saved to {MODELS_DIR}/glove_vectorizer.pkl")
    
    # Evaluate with classifiers
    results = evaluate_vectorizer_with_classifiers(
        X_train_glove, X_val_glove, y_train, y_val, "GloVe", n_cores
    )
    
    # Free memory
    del X_train_glove, X_val_glove, glove_vectorizer
    gc.collect()
    
    return results, {"GloVe": X_train_glove.shape}


def plot_results(all_results, output_path):
    """Create visualization of model performances"""
    
    # Create dataframe from results
    results_df = pd.DataFrame(all_results)
    
    # Plot top models by macro F1
    plt.figure(figsize=(16, 8))
    sns.barplot(x='model_name', y='macro_f1', data=results_df.sort_values('macro_f1', ascending=False).head(15))
    plt.title('Top 15 Models by Macro F1 Score')
    plt.xlabel('Model')
    plt.ylabel('Macro F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_path}/top_15_models.png')
    
    # Get top 5 models
    top_models = results_df.sort_values('macro_f1', ascending=False).head(5)['model_name'].tolist()
    
    # Extract performance by category for top models
    category_results = []
    for result in all_results:
        if result['model_name'] in top_models:
            model_name = result['model_name']
            for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                category_results.append({
                    'model': model_name,
                    'category': col,
                    'f1_score': result[f'f1_{col}']
                })
    
    category_df = pd.DataFrame(category_results)
    
    # Plot performance by category
    plt.figure(figsize=(16, 10))
    sns.barplot(x='category', y='f1_score', hue='model', data=category_df)
    plt.title('F1 Score by Toxicity Category for Top 5 Models')
    plt.xlabel('Toxicity Category')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_path}/category_comparison.png')
    
    print(f"Visualizations saved to {output_path}")
    return results_df


def main():
    """Main function to run all vectorizers and save results"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_data = pd.read_csv('../Dataset/train_preprocessed.csv')
    
    print(f"Training data shape: {train_data.shape}")
    
    # Define features and targets
    X = train_data['processed_text']
    y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    
    # Handle missing values
    X = X.fillna("")
    y = y.fillna(0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['toxic']
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # Dictionary to store all results
    all_classifier_results = []
    feature_shapes = {}
    
    # Define vectorizers to process
    vectorizers = {
        'tfidf': process_tfidf_vectorizer,
        'count': process_count_vectorizer,
        'word2vec': process_word2vec_vectorizer,
        'fasttext': process_fasttext_vectorizer,
        'glove': process_glove_vectorizer
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run text vectorization pipelines')
    parser.add_argument('--vectorizers', nargs='+', choices=vectorizers.keys(), default=list(vectorizers.keys()),
                        help='Select which vectorizers to run')
    args = parser.parse_args()
    
    # Process selected vectorizers
    for name in args.vectorizers:
        print(f"\n\n========== STARTING {name.upper()} VECTORIZATION ==========\n")
        
        # Process this vectorizer
        results, shape = vectorizers[name](X_train, X_val, y_train, y_val)
        
        # Store results and shapes
        all_classifier_results.extend(results)
        feature_shapes.update(shape)
        
        # Clean up after processing
        clean_up_temp_files()
        print(f"\n========== COMPLETED {name.upper()} VECTORIZATION ==========\n")
    
    # Create dataframe from all results
    results_df = plot_results(all_classifier_results, RESULTS_DIR)
    
    # Save results
    results_df.to_csv(f'{RESULTS_DIR}/model_comparison_results.csv', index=False)
    results_df.to_pickle(f'{RESULTS_DIR}/model_comparison_results.pkl')
    print(f"Results saved to {RESULTS_DIR}/model_comparison_results.csv")
    
    # Save feature shapes
    with open(f'{RESULTS_DIR}/feature_shapes.txt', 'w') as f:
        f.write("Feature representation shapes:\n")
        for name, shape in feature_shapes.items():
            f.write(f"{name}: {shape}\n")
    print(f"Feature shapes saved to {RESULTS_DIR}/feature_shapes.txt")
    
    # Identify best classifier for each embedding type
    embedding_types = {}
    for result in all_classifier_results:
        embedding_type = result['model_name'].split(' + ')[0]
        if embedding_type not in embedding_types:
            embedding_types[embedding_type] = []
        embedding_types[embedding_type].append(result)
    
    best_classifiers = {}
    for embedding_type, results in embedding_types.items():
        best_result = max(results, key=lambda x: x['macro_f1'])
        best_classifier_type = best_result['model_name'].split(' + ')[1]
        best_classifiers[embedding_type] = best_classifier_type
        print(f"Best classifier for {embedding_type}: {best_classifier_type}")
    
    # Save best classifier information
    with open(f'{RESULTS_DIR}/best_classifiers.txt', 'w') as f:
        f.write("Best classifier for each embedding type:\n")
        for embedding_type, classifier_type in best_classifiers.items():
            f.write(f"{embedding_type}: {classifier_type}\n")
    
    # Identify overall best model
    best_model = results_df.sort_values('macro_f1', ascending=False).iloc[0]
    print(f"\nThe best performing model overall is: {best_model['model_name']}")
    print(f"  Macro F1: {best_model['macro_f1']:.4f}")
    print(f"  Accuracy: {best_model['accuracy']:.4f}")
    
    # Clean up temporary directory
    clean_up_temp_files()


if __name__ == "__main__":
    main()