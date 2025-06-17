import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results"))
VECTORIZERS_DIR = os.path.join(RESULTS_DIR, "vectorizers_models")
ANALYSIS_PLOTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../analysis_plots"))

def safe_load_pickle(path, name):
    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        print(f"Warning: {name} results file not found at {path}. Skipping.")
        return None

def safe_load_csv(path, name):
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    else:
        print(f"Warning: {name} results file not found at {path}. Skipping.")
        return None

# Load results
rnn_results = safe_load_pickle(os.path.join(RESULTS_DIR, "all_model_results.pkl"), "RNN")
transformer_results = safe_load_csv(os.path.join(RESULTS_DIR, "transformer_model_results.csv"), "Transformer")
vectorizer_results = safe_load_pickle(os.path.join(VECTORIZERS_DIR, "model_comparison_results.pkl"), "Vectorizer")

all_results_list = []

# Prepare RNN results
if rnn_results is not None:
    rnn_results['model_type'] = 'RNN'
    rnn_results = rnn_results.rename(columns={'model_name': 'Model'})
    rnn_results = rnn_results[['Model', 'accuracy', 'macro_f1', 'micro_f1', 'model_type'] + [col for col in rnn_results.columns if col.startswith('f1_')]]
    all_results_list.append(rnn_results)

# Prepare Transformer results
if transformer_results is not None:
    transformer_results = transformer_results.reset_index().rename(columns={'index': 'Model'})
    transformer_results['model_type'] = 'Transformer'
    transformer_results = transformer_results[['Model', 'accuracy', 'macro_f1', 'micro_f1', 'model_type'] + [col for col in transformer_results.columns if col.startswith('f1_')]]
    all_results_list.append(transformer_results)

# Prepare Vectorizer results (take only best for each vectorizer type)
if vectorizer_results is not None:
    vectorizer_results['model_type'] = 'Vectorizer'
    best_vectorizer_models = (
        vectorizer_results.sort_values('macro_f1', ascending=False)
        .groupby(lambda x: vectorizer_results.loc[x, 'model_name'].split(' + ')[0])
        .head(1)
    )
    best_vectorizer_models = best_vectorizer_models.rename(columns={'model_name': 'Model'})
    best_vectorizer_models = best_vectorizer_models[['Model', 'accuracy', 'macro_f1', 'micro_f1', 'model_type'] + [col for col in best_vectorizer_models.columns if col.startswith('f1_')]]
    all_results_list.append(best_vectorizer_models)

if not all_results_list:
    print("No results files found. Please run the model notebooks first.")
else:
    all_results = pd.concat(all_results_list, ignore_index=True)
    all_results = all_results.sort_values('macro_f1', ascending=False)

    # Display combined results
    print("\n=== Overall Model Comparison ===")
    print(all_results[['Model', 'model_type', 'accuracy', 'macro_f1', 'micro_f1']])

    # Plot macro F1 comparison
    plt.figure(figsize=(16, 8))
    sns.barplot(x='Model', y='macro_f1', hue='model_type', data=all_results)
    plt.title('Macro F1 Score Comparison Across Model Families')
    plt.xlabel('Model')
    plt.ylabel('Macro F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    macro_f1_path = os.path.join(ANALYSIS_PLOTS_DIR, "macro_f1_comparison.png")
    plt.savefig(macro_f1_path, bbox_inches='tight')
    plt.show()

    # Detailed per-class F1 comparison
    f1_cols = [col for col in all_results.columns if col.startswith('f1_')]
    melted = all_results.melt(
        id_vars=['Model', 'model_type'],
        value_vars=f1_cols,
        var_name='Class',
        value_name='F1_Score'
    )
    melted['Class'] = melted['Class'].str.replace('f1_', '')

    plt.figure(figsize=(18, 8))
    sns.barplot(x='Class', y='F1_Score', hue='Model', data=melted, dodge=True)
    plt.title('Per-Class F1 Score by Model')
    plt.xlabel('Toxicity Category')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    per_class_f1_path = os.path.join(ANALYSIS_PLOTS_DIR, "per_class_f1_comparison.png")
    plt.savefig(per_class_f1_path, bbox_inches='tight')
    plt.show()

    # Table: Top model per class
    top_per_class = melted.loc[melted.groupby('Class')['F1_Score'].idxmax()]
    print("\n=== Top Model for Each Toxicity Category (by F1) ===")
    print(top_per_class[['Class', 'Model', 'model_type', 'F1_Score']].sort_values('Class'))

    # Table: Best overall model
    print("\n=== Best Overall Model (by Macro F1) ===")
    print(all_results.iloc[0][['Model', 'model_type', 'macro_f1', 'micro_f1', 'accuracy']])