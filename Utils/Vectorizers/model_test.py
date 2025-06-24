import joblib
import pandas as pd
from sklearn.metrics import f1_score
import os
import json
import traceback

# Paths
SAVED_MODELS_DIR = "/work/ssunda2s/toxic_comment_dataset/results/saved_models"
SAVED_DATASETS_DIR = "/work/ssunda2s/toxic_comment_dataset/results/saved_datasets"
test_path = os.path.join(SAVED_DATASETS_DIR, "test.csv")
results_path = os.path.join(SAVED_DATASETS_DIR, "model_test_results.json")
failed_path = os.path.join(SAVED_DATASETS_DIR, "model_test_failed.json")

# Load test data
test_df = pd.read_csv(test_path)
X_test = test_df['processed_text'].fillna("")
y_test = test_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].fillna(0)

# Load previous results if they exist
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        all_results = json.load(f)
    # Build a set of (vectorizer, model) tuples that have already succeeded
    completed = set((r["vectorizer"], r["model"]) for r in all_results)
else:
    all_results = []
    completed = set()

failed_models = []

# Only rerun failed models
if os.path.exists(failed_path):
    with open(failed_path, "r") as f:
        failed_list = json.load(f)
    # Only rerun those not already completed
    model_files = [
        (fail["model_file"], fail["vectorizer_file"])
        for fail in failed_list
        if (fail["model_file"].split("_")[0], "_".join(fail["model_file"].split("_")[1:-1])) not in completed
    ]
else:
    # If no failed file, rerun all
    model_files = [
        (f, f"{f.split('_')[0]}_vectorizer.joblib")
        for f in os.listdir(SAVED_MODELS_DIR) if f.endswith("_model.joblib")
    ]

for model_file, vectorizer_file in model_files:
    vec_name = model_file.split("_")[0]
    model_name = "_".join(model_file.split("_")[1:-1])
    vectorizer_path = os.path.join(SAVED_MODELS_DIR, vectorizer_file)
    model_path = os.path.join(SAVED_MODELS_DIR, model_file)
    if not os.path.exists(vectorizer_path):
        print(f"Vectorizer not found for {model_file}, skipping.")
        failed_models.append({"model_file": model_file, "reason": "vectorizer_not_found"})
        continue

    # Skip if already completed
    if (vec_name, model_name) in completed:
        print(f"Already evaluated {vec_name} + {model_name}, skipping.")
        continue

    print(f"\nEvaluating {vec_name} + {model_name}...")
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        class_f1 = f1_score(y_test, y_pred, average=None)
        result = {
            "vectorizer": vec_name,
            "model": model_name,
            "macro_f1": macro_f1,
            "class_f1": dict(zip(y_test.columns, class_f1))
        }
        all_results.append(result)

        print("Macro F1:", macro_f1)
        print("Class-wise F1:", result["class_f1"])
    except Exception as e:
        print(f"FAILED: {vec_name} + {model_name}: {e}")
        failed_models.append({
            "model_file": model_file,
            "vectorizer_file": vectorizer_file,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# Save all results to JSON
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved all results to {results_path}")

# Save failed models to JSON for rerun
with open(failed_path, "w") as f:
    json.dump(failed_models, f, indent=2)
print(f"Saved failed model info to {failed_path}")