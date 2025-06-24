import re
import pandas as pd

log_file = "/home/ssunda2s/Multi_Label_Toxic_Comment_Classifier/Utils/Vectorizers/logs/vectorizer_eval_559464.log"  # Change to your log filename

results = []
current_combo = None

with open(log_file, "r") as f:
    for line in f:
        # Look for the best combination line
        combo_match = re.search(r"Best combination:\s*(.+?)\s*\+\s*(.+)", line)
        if combo_match:
            current_combo = combo_match.group(1).strip() + " + " + combo_match.group(2).strip()
        # Look for the test F1 score line
        f1_match = re.search(r"Test F1 Score:\s*([0-9.]+)", line)
        if f1_match and current_combo:
            score = float(f1_match.group(1))
            results.append((current_combo, score))
            current_combo = None  # Reset for next block

# Print results
for combo, score in results:
    print(f"{combo}: {score}")

# Optionally, save to CSV
import pandas as pd
df = pd.DataFrame(results, columns=["Combo", "Test_F1_Score"])
df.to_csv("extracted_test_f1_scores.csv", index=False)