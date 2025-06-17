import pandas as pd
import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tqdm import tqdm
import multiprocessing
import yaml
import os
import re
import pickle
import json

# Load configuration
with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'locations.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Initialize Ekphrasis preprocessor
preprocessor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
    fix_html=True, segmenter="twitter", corrector="twitter",
    unpack_hashtags=True, unpack_contractions=True, spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize, dicts=[emoticons]
)

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    
    # Clean problematic text
    if len(text) > 5000:
        text = text[:1000]
    text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
    words = [word[:20] if len(word) > 50 else word for word in text.split()]
    text = ' '.join(words)
    
    if len(text) > 3000:
        return text.lower().strip().split()
    
    # Process tokens
    tokens = preprocessor.pre_process_doc(text)
    processed = []
    
    token_map = {
        '<allcaps>': 'CAPS', '<elongated>': 'EMPHASIS', '<repeated>': 'INTENSE',
        '<user>': 'PERSON', '<url>': 'WEBSITE', '<hashtag>': 'TOPIC'
    }
    
    skip_until = None
    for token in tokens:
        if skip_until:
            if token == skip_until:
                skip_until = None
            elif not token.startswith('<'):
                processed.append(token)
        elif token in token_map:
            processed.append(token_map[token])
            if token in ['<allcaps>', '<elongated>', '<repeated>', '<hashtag>']:
                skip_until = token.replace('<', '</')
        elif not (token.startswith('<') and token.endswith('>')):
            processed.append(token)
    
    return processed

def process_chunk(chunk):
    """Process a chunk of texts - this function can be pickled"""
    return [preprocess_text(text) for text in chunk]

def parallel_preprocess(texts):
    chunk_size = max(1, len(texts) // multiprocessing.cpu_count())
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks),
                           total=len(chunks), desc="Preprocessing"))
    return [item for sublist in results for item in sublist]

def main():
    # Load data
    train_data = pd.read_csv(config['data']['train_raw'])
    test_data = pd.read_csv(config['data']['test_raw'])
    train_data['comment_text'] = train_data['comment_text'].fillna("")
    test_data['comment_text'] = test_data['comment_text'].fillna("")
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Preprocess
    train_tokens = parallel_preprocess(train_data['comment_text'].tolist())
    test_tokens = parallel_preprocess(test_data['comment_text'].tolist())
    
    # Quick stats
    print(f"Avg tokens - Train: {np.mean([len(t) for t in train_tokens]):.1f}, Test: {np.mean([len(t) for t in test_tokens]):.1f}")
    
    # Save
    os.makedirs(os.path.dirname(config['data']['train_tokens']), exist_ok=True)
    
    with open(config['data']['train_tokens'], 'wb') as f:
        pickle.dump({
            'tokens': train_tokens,
            'labels': train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values,
            'ids': train_data['id'].values
        }, f)
    
    with open(config['data']['test_tokens'], 'wb') as f:
        pickle.dump({'tokens': test_tokens, 'ids': test_data['id'].values}, f)
    
    with open(config['data']['train_tokens_sample'], 'w') as f:
        json.dump(train_tokens[:100], f, indent=2)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()