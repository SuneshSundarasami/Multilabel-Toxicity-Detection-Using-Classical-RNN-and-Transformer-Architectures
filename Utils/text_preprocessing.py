import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from typing import Union, Optional, List, Dict
import os

class TextPreprocessor:
    
    def __init__(self, 
                 download_nltk_resources: bool = False,
                 custom_stopwords: Optional[List[str]] = None,
                 keep_negations: bool = True):
        
        if download_nltk_resources:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        self.stop_words = set(stopwords.words('english'))
        
        if keep_negations:
            negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nowhere', 'nothing', 'nor', 'neither'}
            self.stop_words = self.stop_words - negation_words
        
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        if isinstance(text, str):
            text = text.lower()
            
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            text = re.sub(r'<.*?>', '', text)
            
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        else:
            return ""
    
    def remove_stopwords(self, text: str) -> str:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)
    
    def stem_text(self, text: str) -> str:
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)
    
    def preprocess_text(self, 
                         text: str, 
                         clean: bool = True,
                         remove_stops: bool = True, 
                         lemmatize: bool = True, 
                         stem: bool = False) -> str:
        
        if clean:
            text = self.clean_text(text)
        
        if remove_stops:
            text = self.remove_stopwords(text)
        
        if lemmatize:
            text = self.lemmatize_text(text)
        
        if stem:
            text = self.stem_text(text)
        
        return text
    
    def preprocess_dataframe(self, 
                             df: pd.DataFrame, 
                             text_column: str,
                             output_column: str = 'processed_text',
                             clean: bool = True,
                             remove_stops: bool = True,
                             lemmatize: bool = True,
                             stem: bool = False,
                             inplace: bool = False) -> pd.DataFrame:
        
        if not inplace:
            df = df.copy()
        
        df[text_column] = df[text_column].fillna("")
        
        df[output_column] = df[text_column].apply(
            lambda x: self.preprocess_text(
                x, 
                clean=clean,
                remove_stops=remove_stops, 
                lemmatize=lemmatize, 
                stem=stem
            )
        )
        
        return df
    
    def preprocess_dataset(self,
                           train_path: str,
                           test_path: str,
                           text_column: str = 'comment_text',
                           output_column: str = 'processed_text',
                           test_labels_path: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           clean: bool = True,
                           remove_stops: bool = True,
                           lemmatize: bool = True,
                           stem: bool = False) -> Dict[str, pd.DataFrame]:
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        if output_dir is None:
            output_dir = 'Dataset'
        
        if test_labels_path and os.path.exists(test_labels_path):
            test_labels = pd.read_csv(test_labels_path)
            if 'id' in test_data.columns and 'id' in test_labels.columns:
                test_data = test_data.merge(test_labels, on='id', how='left')
        
        print("Preprocessing training data...")
        train_processed = self.preprocess_dataframe(
            train_data,
            text_column=text_column,
            output_column=output_column,
            clean=clean,
            remove_stops=remove_stops,
            lemmatize=lemmatize,
            stem=stem
        )
        
        print("Preprocessing test data...")
        test_processed = self.preprocess_dataframe(
            test_data,
            text_column=text_column,
            output_column=output_column,
            clean=clean,
            remove_stops=remove_stops,
            lemmatize=lemmatize,
            stem=stem
        )
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            train_output_path = os.path.join(output_dir, 'train_preprocessed.csv')
            test_output_path = os.path.join(output_dir, 'test_preprocessed.csv')
            
            train_processed.to_csv(train_output_path, index=False)
            test_processed.to_csv(test_output_path, index=False)
            print(f"Preprocessed data saved to {output_dir}")
        
        return {
            'train': train_processed,
            'test': test_processed
        }

    def analyze_preprocessing_effect(self, 
                                    original_df: pd.DataFrame, 
                                    processed_df: pd.DataFrame,
                                    text_column: str = 'comment_text',
                                    processed_column: str = 'processed_text',
                                    plot: bool = True,
                                    output_dir: str = 'analysis_plots') -> Dict:
        
        processed_df['original_length'] = processed_df[text_column].apply(len)
        processed_df['processed_length'] = processed_df[processed_column].apply(len)
        processed_df['length_reduction'] = (1 - processed_df['processed_length'] / processed_df['original_length']) * 100
        
        avg_reduction = processed_df['length_reduction'].mean()
        print(f"Average text length reduction: {avg_reduction:.2f}%")
        
        def get_vocabulary_size(texts):
            all_words = ' '.join(texts).split()
            unique_words = set(all_words)
            return len(unique_words)
        
        original_vocab_size = get_vocabulary_size(original_df[text_column])
        processed_vocab_size = get_vocabulary_size(processed_df[processed_column])
        vocab_reduction = (1 - processed_vocab_size / original_vocab_size) * 100
        
        print(f"Original vocabulary size: {original_vocab_size:,}")
        print(f"Processed vocabulary size: {processed_vocab_size:,}")
        print(f"Vocabulary reduction: {vocab_reduction:.2f}%")
        
        if plot:
            import matplotlib.pyplot as plt
            from collections import Counter
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            plt.figure(figsize=(10, 6))
            plt.hist(processed_df['length_reduction'], bins=50)
            plt.axvline(avg_reduction, color='r', linestyle='--', 
                      label=f'Mean reduction: {avg_reduction:.2f}%')
            plt.xlabel('Length Reduction (%)')
            plt.ylabel('Number of Comments')
            plt.title('Text Length Reduction after Preprocessing')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'length_reduction_histogram.png'))
            plt.close()
            
            all_words = ' '.join(processed_df[processed_column]).split()
            word_counts = Counter(all_words).most_common(20)
            words, counts = zip(*word_counts)
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(words)), counts, align='center')
            plt.yticks(range(len(words)), words)
            plt.title('Top 20 Most Common Words After Preprocessing')
            plt.xlabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_words_frequency.png'))
            plt.close()
            
            print(f"Analysis plots saved to {output_dir}")
        
        return {
            'length_reduction': avg_reduction,
            'vocabulary_size': {
                'original': original_vocab_size,
                'processed': processed_vocab_size,
                'reduction_pct': vocab_reduction
            },
            'processed_df': processed_df
        }

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor(download_nltk_resources=True)
    
    # Check if input files exist
    train_path = 'Dataset/train.csv'
    test_path = 'Dataset/test.csv'
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Starting text preprocessing...")
        result = preprocessor.preprocess_dataset(
            train_path=train_path,
            test_path=test_path
        )
        print("Preprocessing completed!")
        print(f"Train data shape: {result['train'].shape}")
        print(f"Test data shape: {result['test'].shape}")
        print(f"Train data head:\n{result['train'].head()}")
        print(f"Test data head:\n{result['test'].head()}")
    else:
        print(f"Input files not found!")
        print(f"Looking for: {train_path}, {test_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")