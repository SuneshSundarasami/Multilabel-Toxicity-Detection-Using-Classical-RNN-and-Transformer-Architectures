import numpy as np
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec, FastText

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    # ... (your code) ...

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    # ... (your code) ...

class GloveVectorizer(BaseEstimator, TransformerMixin):
    # ... (your code) ...