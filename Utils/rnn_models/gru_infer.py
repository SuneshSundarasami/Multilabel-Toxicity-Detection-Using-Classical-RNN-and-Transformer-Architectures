import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pickle
import numpy as np

from Utils.rnn_models.rnn_models import load_config, load_trained_model
from Utils.preprocessing.preprocessor import preprocess_text

# --- CONFIGURATION ---
CONFIG = load_config()
MODEL_NAME = CONFIG['rnn_models'].get('model_name', 'bilstm_attention')
MODEL_PATH = os.path.join(CONFIG['rnn_models']['root_dir'], f"models/{MODEL_NAME}_best.pt")
VOCAB_PATH = os.path.join(CONFIG['rnn_models']['root_dir'], "models/vocabulary.pkl")
EMBEDDING_MATRIX_PATH = os.path.join(CONFIG['rnn_models']['root_dir'], "models/embedding_matrix.npy")
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
N_LAYERS = 3
DROPOUT = 0.5
NUM_CLASSES = 6

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def preprocess_and_index(text, word2idx):
    tokens = preprocess_text(text)
    indices = [word2idx.get(tok, word2idx.get('<unk>', 0)) for tok in tokens]
    if not indices:
        indices = [word2idx.get('<unk>', 0)]
    print(f"Preprocessed tokens: {tokens}")
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

def main():
    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    word2idx = vocab['word2idx']

    # Load embedding matrix
    if os.path.exists(EMBEDDING_MATRIX_PATH):
        embedding_matrix = np.load(EMBEDDING_MATRIX_PATH)
        print(f"Loaded embedding matrix of shape: {embedding_matrix.shape}")
    else:
        embedding_matrix = None
        print("Embedding matrix not found, using random initialization.")

    # Load model using the new utility function
    model = load_trained_model(
        MODEL_NAME,
        MODEL_PATH,
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=NUM_CLASSES,
        pretrained_embeddings=embedding_matrix,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        device='cpu'
    )

    # User input
    text = "I am proud to be a straight."
    indices = preprocess_and_index(text, word2idx)
    
    with torch.no_grad():
        probs = model(indices).squeeze().cpu().numpy()
        preds = (probs > 0.5).astype(int)

    print("\nClassification results:")
    for label, prob, pred in zip(LABELS, probs, preds):
        print(f"{label:15s}: {prob:.3f} ({'YES' if pred else 'NO'})")

if __name__ == "__main__":
    main()