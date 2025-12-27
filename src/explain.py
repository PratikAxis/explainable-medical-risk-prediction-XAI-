import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PATH, MODEL_PATH


# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("HeartDisease", axis=1)

# Load trained model
model = joblib.load(MODEL_PATH)

# Transform features
X_transformed = model.named_steps["preprocessor"].transform(X)

# Get feature names after preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# SHAP explainer (Logistic Regression → LinearExplainer)
explainer = shap.LinearExplainer(
    model.named_steps["classifier"],
    X_transformed,
    feature_names=feature_names
)

shap_values = explainer(X_transformed)

# Plot
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)

# ---------- TEXT EXPLANATION ----------
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:3]

print("\nTop 3 Important Medical Risk Factors:\n")

for i in top_idx:
    print(
        f"- {feature_names[i]} "
        f"(mean |SHAP| = {mean_abs_shap[i]:.4f})"
    )



# ---------- SAVE EXPLANATION TEXT ----------
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

report_path = REPORT_DIR / "shap_explanation.txt"

lines = []
lines.append("MODEL EXPLANATION USING SHAP\n")
lines.append("--------------------------------------------------\n")

lines.append(
    "This explanation summarizes the most influential features "
    "affecting the heart disease prediction model.\n\n"
)

for rank, i in enumerate(top_idx, start=1):
    feature = feature_names[i]
    impact = mean_abs_shap[i]

    direction = (
        "increases" if shap_values.values[:, i].mean() > 0
        else "decreases"
    )

    lines.append(
        f"{rank}. {feature} has a strong impact on the prediction. "
        f"Higher values of this feature generally {direction} "
        f"the likelihood of heart disease "
        f"(mean |SHAP| = {impact:.3f}).\n"
    )

lines.append(
    "\nOverall, the model relies primarily on clinical measurements "
    "and exercise-related indicators to assess cardiovascular risk. "
    "This improves transparency and trustworthiness of the prediction system.\n"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nExplanation text saved to: {report_path}")
