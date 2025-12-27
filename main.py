"""
Main entry point for Medical Risk Prediction with Explainable AI
"""

print("\n========== MEDICAL RISK PREDICTION XAI ==========\n")

# Step 1: Train model
print("Step 1: Training model...")
import src.train
print("[OK] Model training completed\n")

# Step 2: Explain model
print("Step 2: Generating explanations...")
import src.explain
print("[OK] SHAP explanations generated\n")

print("========== PIPELINE COMPLETED SUCCESSFULLY ==========\n")
