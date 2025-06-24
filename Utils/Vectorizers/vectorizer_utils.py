import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from joblib import Parallel, delayed
import psutil
import torch
from math import ceil

# Get number of CPU cores
N_CORES = psutil.cpu_count(logical=True)
print(f"Using {N_CORES} CPU cores for parallel processing")

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=300):
        self.vector_size = vector_size
        self.model = None
    
    def fit(self, X, y=None):
        tokenized = [text.lower().split() for text in X]
        self.model = Word2Vec(tokenized, vector_size=self.vector_size, min_count=1, workers=N_CORES)
        return self
    
    def transform(self, X):
        def vectorize_text(text):
            tokens = text.lower().split()
            return np.mean([self.model.wv[token] for token in tokens if token in self.model.wv] or [np.zeros(self.vector_size)], axis=0)
        
        vectors = Parallel(n_jobs=N_CORES, backend='threading')(
            delayed(vectorize_text)(text) for text in X
        )
        return np.array(vectors)

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=300):
        self.vector_size = vector_size
        self.model = None
    
    def fit(self, X, y=None):
        tokenized = [text.lower().split() for text in X]
        self.model = FastText(tokenized, vector_size=self.vector_size, min_count=1, workers=N_CORES)
        return self
    
    def transform(self, X):
        def vectorize_text(text):
            tokens = text.lower().split()
            return np.mean([self.model.wv[token] for token in tokens], axis=0) if tokens else np.zeros(self.vector_size)
        
        vectors = Parallel(n_jobs=N_CORES, backend='threading')(
            delayed(vectorize_text)(text) for text in X
        )
        return np.array(vectors)

class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.word_vectors = {}
    
    def fit(self, X, y=None):
        try:
            glove_model = api.load("glove-wiki-gigaword-100")
            self.word_vectors = {word: glove_model[word] for word in glove_model.key_to_index}
        except:
            self.word_vectors = {}
        return self
    
    def transform(self, X):
        def vectorize_text(text):
            tokens = text.lower().split()
            return np.mean([self.word_vectors[token] for token in tokens if token in self.word_vectors] or [np.zeros(self.vector_size)], axis=0)
        
        vectors = Parallel(n_jobs=N_CORES, backend='threading')(
            delayed(vectorize_text)(text) for text in X
        )
        return np.array(vectors)

class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name, device=DEVICE)
        self.model.max_seq_length = 512
        return self

    def _encode_chunk(self, texts):
        # Each process loads its own model instance
        model = SentenceTransformer(self.model_name, device=DEVICE)
        model.max_seq_length = 512
        # Use a reasonable batch size for your RAM
        batch_size = 256 if DEVICE == 'cuda' else 64
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            device=DEVICE,
            normalize_embeddings=True
        )

    def transform(self, X):
        X = X.tolist()
        n_chunks = N_CORES
        chunk_size = ceil(len(X) / n_chunks)
        chunks = [X[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
        # Parallel encoding
        results = Parallel(n_jobs=N_CORES)(
            delayed(self._encode_chunk)(chunk) for chunk in chunks if chunk
        )
        # Concatenate results
        return np.vstack(results)

def get_vectorizers():
    """Return dictionary of all available vectorizers including sentence transformers"""
    return {
        # Traditional vectorizers
        'TF-IDF': TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
        'Count': CountVectorizer(max_features=20000, ngram_range=(1,2)),
        'Word2Vec': Word2VecVectorizer(),
        'FastText': FastTextVectorizer(),
        'GloVe': GloveVectorizer(),
        
        # Top 3 sentence transformers
        'SentenceTransformer-MiniLM': SentenceTransformerVectorizer('all-MiniLM-L6-v2'),
        'SentenceTransformer-MPNet': SentenceTransformerVectorizer('all-mpnet-base-v2'),
        'SentenceTransformer-BGE-Large': SentenceTransformerVectorizer('BAAI/bge-large-en-v1.5')
    }

def objective_logistic_regression(trial, X_train_vec, y_train, X_val_vec, y_val, penalty='l2'):
    C = trial.suggest_float('C', 0.01, 100, log=True)
    max_iter = trial.suggest_int('max_iter', 2000, 10000)  # Increased significantly
    
    # Convert data to float32 for consistency
    if hasattr(X_train_vec, 'astype'):
        X_train_vec = X_train_vec.astype(np.float32)
    if hasattr(X_val_vec, 'astype'):
        X_val_vec = X_val_vec.astype(np.float32)
    
    model = MultiOutputClassifier(
        LogisticRegression(
            C=C, 
            penalty=penalty, 
            max_iter=max_iter, 
            random_state=42,
            solver='liblinear',  # More robust for sparse data
            n_jobs=1
        ),
        n_jobs=N_CORES
    )
    
    try:
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_val_vec)
        
        f1_scores = [
            f1_score(y_val.iloc[:, i], y_pred[:, i], average='macro')
            for i in range(y_val.shape[1])
        ]
        
        return np.mean(f1_scores)
    except Exception as e:
        print(f"LogisticRegression failed: {e}")
        return 0.0

def objective_xgboost(trial, X_train_vec, y_train, X_val_vec, y_val):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    xgb_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    if DEVICE == 'cuda':
        xgb_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        })
        multioutput_n_jobs = 1
    else:
        xgb_params.update({
            'tree_method': 'hist',
            'n_jobs': N_CORES
        })
        multioutput_n_jobs = N_CORES
    
    model = MultiOutputClassifier(
        xgb.XGBClassifier(**xgb_params), 
        n_jobs=multioutput_n_jobs
    )
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_val_vec)
    
    def compute_f1(i):
        return f1_score(y_val.iloc[:, i], y_pred[:, i], average='macro')
    
    f1_scores = Parallel(n_jobs=N_CORES)(
        delayed(compute_f1)(i) for i in range(y_val.shape[1])
    )
    
    return np.mean(f1_scores)

def objective_lightgbm(trial, X_train_vec, y_train, X_val_vec, y_val):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    # Convert sparse matrices to float32 for LightGBM compatibility
    if hasattr(X_train_vec, 'dtype') and X_train_vec.dtype != np.float32:
        X_train_vec = X_train_vec.astype(np.float32)
    if hasattr(X_val_vec, 'dtype') and X_val_vec.dtype != np.float32:
        X_val_vec = X_val_vec.astype(np.float32)
    
    lgb_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': 42,
        'verbose': -1,
        'class_weight': 'balanced'
    }
    
    if DEVICE == 'cuda':
        lgb_params.update({
            'device': 'gpu',
            'objective': 'binary'
        })
        multioutput_n_jobs = 1
    else:
        lgb_params.update({
            'device': 'cpu',
            'n_jobs': N_CORES
        })
        multioutput_n_jobs = N_CORES
    
    model = MultiOutputClassifier(
        lgb.LGBMClassifier(**lgb_params),
        n_jobs=multioutput_n_jobs
    )
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_val_vec)
    
    def compute_f1(i):
        return f1_score(y_val.iloc[:, i], y_pred[:, i], average='macro')
    
    f1_scores = Parallel(n_jobs=N_CORES)(
        delayed(compute_f1)(i) for i in range(y_val.shape[1])
    )
    
    return np.mean(f1_scores)

def get_models():
    """Return dictionary of all available model objective functions"""
    return {
        # 'LogisticRegression_L1': lambda trial, X_tr, y_tr, X_val, y_val: objective_logistic_regression(trial, X_tr, y_tr, X_val, y_val, 'l1'),
        'LogisticRegression_L2': lambda trial, X_tr, y_tr, X_val, y_val: objective_logistic_regression(trial, X_tr, y_tr, X_val, y_val, 'l2'),
        'XGBoost': objective_xgboost,
        'LightGBM': objective_lightgbm
    }

def rebuild_best_model(model_name, best_params_str, X_trainval_vec, y_trainval, X_test_vec, y_test,return_model=False):
    """Rebuild and evaluate the best model on test set"""
    import ast
    
    try:
        best_params = ast.literal_eval(best_params_str)
    except:
        print("Error parsing best parameters")
        return 0.0
    
    if model_name == 'LogisticRegression_L1':
        model = MultiOutputClassifier(LogisticRegression(
            penalty='l1',
            solver='liblinear',  # <-- fastest and correct for L1
            random_state=42,
            class_weight='balanced',
            **best_params
        ), n_jobs=N_CORES)
    elif model_name == 'LogisticRegression_L2':
        model = MultiOutputClassifier(LogisticRegression(
            penalty='l2',
            solver='lbfgs',      # <-- fastest for large/dense, or use 'liblinear' for sparse
            random_state=42,
            class_weight='balanced',
            **best_params
        ), n_jobs=N_CORES)
    elif model_name == 'XGBoost':
        xgb_params = {**best_params, 'random_state': 42, 'eval_metric': 'logloss'}
        xgb_params['scale_pos_weight'] = compute_scale_pos_weight(y_trainval)
        xgb_params.pop('tree_method', None)
        xgb_params.pop('gpu_id', None)
        xgb_params.pop('device', None)
        if DEVICE == 'cuda':
            xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
            n_jobs = 1
        else:
            xgb_params.update({'tree_method': 'hist', 'n_jobs': N_CORES})
            n_jobs = N_CORES
        model = MultiOutputClassifier(xgb.XGBClassifier(**xgb_params), n_jobs=n_jobs)
    elif model_name == 'LightGBM':
        lgb_params = {**best_params, 'random_state': 42, 'verbose': -1, 'class_weight': 'balanced'}
        if DEVICE == 'cuda':
            lgb_params.update({'device': 'gpu', 'objective': 'binary'})
            n_jobs = 1
        else:
            lgb_params.update({'device': 'cpu', 'n_jobs': N_CORES})
            n_jobs = N_CORES
        model = MultiOutputClassifier(lgb.LGBMClassifier(**lgb_params), n_jobs=n_jobs)
    else:
        print(f"Unknown model: {model_name}")
        return 0.0
    
    model.fit(X_trainval_vec, y_trainval)
    if return_model:
        return model
    y_pred = model.predict(X_test_vec)
    
    def compute_f1(i):
        return f1_score(y_test.iloc[:, i], y_pred[:, i], average='macro')
    
    f1_scores = Parallel(n_jobs=N_CORES)(
        delayed(compute_f1)(i) for i in range(y_test.shape[1])
    )
    
    return np.mean(f1_scores)

def compute_scale_pos_weight(y):
    # y is a DataFrame (multi-label), return average pos_weight
    weights = []
    for col in y.columns:
        n_pos = y[col].sum()
        n_neg = len(y[col]) - n_pos
        if n_pos == 0:
            weights.append(1.0)
        else:
            weights.append(n_neg / n_pos)
    return float(np.mean(weights))