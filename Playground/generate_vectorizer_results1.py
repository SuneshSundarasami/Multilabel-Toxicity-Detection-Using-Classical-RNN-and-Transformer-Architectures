import pandas as pd
import pickle
import os
import nltk
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec, FastText

# ---- Custom Vectorizer Classes ----
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
        self.word_vectors = None

    def fit(self, X, y=None):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in X]
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        self.word_vectors = self.model.wv
        return self

    def transform(self, X):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in X]
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
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in X]
        self.model = FastText(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        return self

    def transform(self, X):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in X]
        doc_vectors = np.zeros((len(tokenized_corpus), self.vector_size))
        for i, tokens in enumerate(tokenized_corpus):
            vec = np.zeros(self.vector_size)
            count = 0
            for token in tokens:
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
        import gensim.downloader as api
        glove_model = api.load("glove-wiki-gigaword-100")
        self.word_vectors = {word: glove_model[word] for word in glove_model.key_to_index}
        return self

    def transform(self, X):
        tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in X]
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

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../Dataset/train_preprocessed.csv")
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results"))
VECTORIZERS_DIR = os.path.join(RESULTS_DIR, "vectorizers_models")
os.makedirs(VECTORIZERS_DIR, exist_ok=True)

# ---- Load data ----
train_data = pd.read_csv(DATA_PATH)
X = train_data['processed_text'].fillna("")
y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].fillna(0)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['toxic']
)

# ---- Load vectorizers ----
vectorizer_names = ['tfidf', 'count', 'word2vec', 'fasttext', 'glove']
vectorizers = {}
for name in vectorizer_names:
    path = os.path.join(BASE_DIR, "../models", f"{name}_vectorizer.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            vectorizers[name] = pickle.load(f)
    else:
        print(f"Vectorizer {name} not found at {path}")

# ---- Transform data ----
X_train_transformed = {}
X_val_transformed = {}
for name, vect in vectorizers.items():
    print(f"Transforming training data with {name}...")
    X_train_transformed[name] = vect.transform(X_train)
    print(f"Transforming validation data with {name}...")
    X_val_transformed[name] = vect.transform(X_val)

# ---- Load best classifiers for each embedding ----
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score

best_classifiers = {
    'tfidf': MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')),
    'count': MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')),
    'word2vec': MultiOutputClassifier(LinearSVC(max_iter=10000, class_weight='balanced', dual=False)),
    'fasttext': MultiOutputClassifier(LinearSVC(max_iter=10000, class_weight='balanced', dual=False)),
    'glove': MultiOutputClassifier(LinearSVC(max_iter=10000, class_weight='balanced', dual=False)),
}

# ---- Train, predict, and collect results ----
results = []
for name in vectorizer_names:
    if name in best_classifiers and name in X_train_transformed and name in X_val_transformed:
        clf = best_classifiers[name]
        print(f"Training {name} classifier...")
        clf.fit(X_train_transformed[name], y_train)
        print(f"Predicting with {name} classifier...")
        y_pred = clf.predict(X_val_transformed[name])
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        micro_f1 = f1_score(y_val, y_pred, average='micro')
        result = {
            'model_name': f"{name} + {clf.estimator.__class__.__name__}",
            'accuracy': acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        }
        for i, col in enumerate(y_val.columns):
            result[f'f1_{col}'] = f1_score(y_val[col], y_pred[:, i])
        results.append(result)

results_df = pd.DataFrame(results)
results_df.to_pickle(os.path.join(VECTORIZERS_DIR, "model_comparison_results.pkl"))
print("Saved vectorizer validation results to model_comparison_results.pkl")