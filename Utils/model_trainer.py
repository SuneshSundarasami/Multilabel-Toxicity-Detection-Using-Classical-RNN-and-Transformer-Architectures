import pandas as pd
import numpy as np
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

class ModelTrainer:
    def __init__(self,
                 model_dir: str = 'models'):
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def prepare_features(self,
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         text_column: str = 'processed_text',
                         vectorizer_type: str = 'tfidf',
                         max_features: int = 10000,
                         ngram_range: Tuple[int, int] = (1, 2)) -> Dict[str, Any]:
        # Only consider columns that are likely to be label columns
        # Exclude text columns and metadata columns
        target_columns = [col for col in train_data.columns 
                          if col not in [text_column, 'comment_text', 'id', 'original_length', 
                                        'processed_length', 'length_reduction']]
        
        print(f"Target columns: {target_columns}")
        
        # Ensure text column doesn't have NaN values
        train_data[text_column] = train_data[text_column].fillna("")
        test_data[text_column] = test_data[text_column].fillna("")
        
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features, 
                                        ngram_range=ngram_range)
        else:
            vectorizer = CountVectorizer(max_features=max_features, 
                                        ngram_range=ngram_range)
        
        X_train = vectorizer.fit_transform(train_data[text_column])
        X_test = vectorizer.transform(test_data[text_column])
        
        y_train = train_data[target_columns].values
        y_test = test_data[target_columns].values
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'vectorizer': vectorizer,
            'target_columns': target_columns
        }
    
    def train_model(self,
                   data: Dict[str, Any],
                   model_type: str = 'logistic',
                   model_params: Optional[Dict] = None,
                   search_params: bool = False,
                   cv: int = 3) -> Dict[str, Any]:
        X_train = data['X_train']
        y_train = data['y_train']
        
        if model_params is None:
            model_params = {}
        
        if model_type == 'logistic':
            base_model = LogisticRegression(**model_params)
            param_grid = {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__solver': ['lbfgs', 'liblinear'],
                'estimator__class_weight': [None, 'balanced']
            }
        elif model_type == 'svm':
            base_model = LinearSVC(**model_params)
            param_grid = {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__class_weight': [None, 'balanced']
            }
        elif model_type == 'rf':
            base_model = RandomForestClassifier(**model_params)
            param_grid = {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5, 10]
            }
        elif model_type == 'nb':
            base_model = MultinomialNB(**model_params)
            param_grid = {
                'estimator__alpha': [0.01, 0.1, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = MultiOutputClassifier(base_model)
        
        if search_params:
            print(f"Performing grid search for {model_type} model...")
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            print(f"Best parameters: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
        
        print(f"Model training completed in {train_time:.2f} seconds")
        
        model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'vectorizer': data['vectorizer'],
                'target_columns': data['target_columns']
            }, f)
        
        print(f"Model saved to {model_path}")
        
        return {
            'model': model,
            'train_time': train_time,
            'model_path': model_path
        }

    def predict(self,
               model_path: str,
               texts: List[str],
               threshold: float = 0.5) -> pd.DataFrame:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        target_columns = model_data['target_columns']
        
        X = vectorizer.transform(texts)
        
        y_pred_proba = model.predict_proba(X)
        
        results = pd.DataFrame()
        
        for i, col in enumerate(target_columns):
            results[f"{col}_probability"] = y_pred_proba[i][:, 1]
            results[col] = (y_pred_proba[i][:, 1] >= threshold).astype(int)
        
        results['text'] = texts
        
        return results