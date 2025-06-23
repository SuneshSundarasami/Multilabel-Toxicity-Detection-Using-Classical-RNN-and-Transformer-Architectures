import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from pandas.plotting import parallel_coordinates
import plotly.express as px

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

# Prepare Vectorizer results (from pickle and JSON)
vectorizer_dfs = []
if vectorizer_results is not None:
    vectorizer_results['model_type'] = 'Vectorizer'
    best_vectorizer_models = (
        vectorizer_results.sort_values('macro_f1', ascending=False)
        .groupby(lambda x: vectorizer_results.loc[x, 'model_name'].split(' + ')[0])
        .head(1)
    )
    best_vectorizer_models = best_vectorizer_models.rename(columns={'model_name': 'Model'})
    best_vectorizer_models = best_vectorizer_models[['Model', 'accuracy', 'macro_f1', 'micro_f1', 'model_type'] + [col for col in best_vectorizer_models.columns if col.startswith('f1_')]]
    vectorizer_dfs.append(best_vectorizer_models)

json_path = "/home/sunesh/NLP/Multi_Label_Toxic_Comment_Classifier/Utils/Vectorizers/results/model_results.json"
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    filtered = [
        entry for entry in json_data
        if entry and 'macro_f1' in entry and 'vectorizer' in entry and 'model' in entry
    ]
    if filtered:
        for entry in filtered:
            for k, v in entry.get("class_f1", {}).items():
                entry[f"f1_{k}"] = v
        json_df = pd.DataFrame(filtered)
        json_df['Model'] = json_df['vectorizer'] + " + " + json_df['model']
        json_df['model_type'] = 'Vectorizer'
        for col in ['accuracy', 'micro_f1']:
            if col not in json_df.columns:
                json_df[col] = float('nan')
        json_df = json_df[['Model', 'accuracy', 'macro_f1', 'micro_f1', 'model_type'] + [col for col in json_df.columns if col.startswith('f1_')]]
        vectorizer_dfs.append(json_df)

# Combine all vectorizer results and select top 4 by macro_f1
if vectorizer_dfs:
    all_vectorizer_results = pd.concat(vectorizer_dfs, ignore_index=True)
    top4_vectorizer = all_vectorizer_results.sort_values('macro_f1', ascending=False).head(4)
    all_results_list.append(all_vectorizer_results)  # or top4_vectorizer if you want only top 4

# Add GPT1 model results manually
gpt1_results = pd.DataFrame([{
    'Model': 'GPT1',
    'accuracy': 0.912,
    'macro_f1': 0.606,
    'micro_f1': 0.980,
    'model_type': 'Transformer',
    'f1_toxic': 0.824,
    'f1_severe_toxic': 0.703,
    'f1_obscene': 0.826,
    'f1_threat': 0.376,
    'f1_insult': 0.817,
    'f1_identity_hate': 0.426
}])

flant5base_results = pd.DataFrame([{
    'Model': 'FLANT5BASE',
    'accuracy': 0.909,
    'macro_f1': 0.574,
    'micro_f1': 0.979,
    'model_type': 'Transformer',
    'f1_toxic': 0.817,
    'f1_severe_toxic': 0.462,
    'f1_obscene': 0.814,
    'f1_threat': 0.225,
    'f1_insult': 0.733,
    'f1_identity_hate': 0.391
}])

all_results_list.append(gpt1_results)
all_results_list.append(flant5base_results)

if not all_results_list:
    print("No results files found. Please run the model notebooks first.")
    exit()

all_results = pd.concat(all_results_list, ignore_index=True)
all_results = all_results.sort_values('macro_f1', ascending=False)

# --- Ensure SentenceTransformer-LGBM and BERT-LGBM are in Top 10 Vectorizer Models ---

# Find top 10 by macro_f1
top10 = all_results.sort_values('macro_f1', ascending=False).head(10).copy()

# Identify the two special models from model_results.json
special_models = [
    "SentenceTransformer-MiniLM-L6-v2 + LightGBM",
    "BERT-base-uncased + LightGBM"
]

# Check if they are present
present = top10['Model'].tolist()
missing = [m for m in special_models if m not in present]

# If missing, try to find them in all_results and replace the last ones
if missing:
    # Find their rows in all_results
    special_rows = all_results[all_results['Model'].isin(missing)]
    # Remove as many from the end of top10 as needed
    if not special_rows.empty:
        # Remove last N rows from top10 (N = len(special_rows))
        top10 = top10.iloc[:-len(special_rows)]
        # Append the special models
        top10 = pd.concat([top10, special_rows], ignore_index=True)

# Now use top10 for your plotting
sorted_results = top10.sort_values('macro_f1', ascending=False).reset_index(drop=True)

# --- Visualization ---

# --- Use only the models selected for the final (top10) graph in all visualizations ---

# 2. Per-Class F1 Score by Model (Grouped Bar Chart, restricted to top10 models)
f1_cols = [col for col in all_results.columns if col.startswith('f1_')]
melted = all_results[all_results['Model'].isin(sorted_results['Model'])].melt(
    id_vars=['Model', 'model_type'],
    value_vars=f1_cols,
    var_name='Class',
    value_name='F1_Score'
)
melted['Class'] = melted['Class'].str.replace('f1_', '')

plt.figure(figsize=(18, 8))
ax2 = sns.barplot(
    x='Class', y='F1_Score', hue='Model', data=melted,
    dodge=True, edgecolor='black'
)
plt.title('How Well Do Top Models Detect Each Toxicity Category?', fontsize=18, weight='bold', loc='center')
plt.xlabel('Toxicity Category', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
per_class_f1_path = os.path.join(ANALYSIS_PLOTS_DIR, "per_class_f1_comparison.png")
plt.savefig(per_class_f1_path, bbox_inches='tight', dpi=300)  # High resolution
plt.show()

# 3. Table: Top model per class
top_per_class = melted.loc[melted.groupby('Class')['F1_Score'].idxmax()]
print("\n=== Top Model for Each Toxicity Category (by F1) ===")
print(top_per_class[['Class', 'Model', 'model_type', 'F1_Score']].sort_values('Class'))

# 4. Callout: Best overall model
best = all_results.iloc[0]
print("\n=== Best Overall Model (by Macro F1) ===")
print(best[['Model', 'model_type', 'macro_f1', 'micro_f1', 'accuracy']])
print(f"\n🏆 The best overall model is **{best['Model']}** ({best['model_type']}) with a Macro F1 of {best['macro_f1']:.2f}.")

# --- Fancy Macro F1 Score Comparison (Horizontal Bar Chart with In-Bar Labels, Best F1 at Top) ---
plt.figure(figsize=(12, max(6, 0.6 * len(sorted_results))))
palette = {"RNN": "#1f77b4", "Transformer": "#ff7f0e", "Vectorizer": "#2ca02c"}
sorted_results = top10.sort_values('macro_f1', ascending=False).reset_index(drop=True)
bar_colors = [palette[mt] for mt in sorted_results['model_type']]

ax = sns.barplot(
    y='Model', x='macro_f1', data=sorted_results,
    palette=bar_colors, edgecolor='black'
)

plt.title('Top 10 Models: Overall Macro F1 Score Leaderboard', fontsize=18, weight='bold', pad=15, loc='center')
plt.xlabel('Macro F1 Score', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.xlim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for i, p in enumerate(ax.patches):
    width = p.get_width()
    y = p.get_y() + p.get_height() / 2
    model_type = sorted_results.iloc[i]['model_type']
    color = palette[model_type]
    ax.annotate(f"{width:.2f}", (width + 0.01, y),
                ha='left', va='center', fontsize=11, color='black', weight='bold')
    ax.annotate(model_type, (min(width * 0.7, 0.85), y),
                ha='center', va='center', fontsize=10, color='white', weight='bold',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))

plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
macro_f1_path = os.path.join(ANALYSIS_PLOTS_DIR, "macro_f1_comparison_1.png")
plt.savefig(macro_f1_path, bbox_inches='tight', dpi=300)  # High resolution
plt.show()
