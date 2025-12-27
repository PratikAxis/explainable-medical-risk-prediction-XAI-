from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.preprocessing import preprocessor   # make sure preprocessing.py defines this

def build_model():
    """
    Build and return the ML pipeline for heart disease prediction.
    The pipeline includes preprocessing (scaling + encoding) and a logistic regression classifier.
    """
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,          # allow more iterations for convergence
            class_weight="balanced",# handle class imbalance
            solver="liblinear"      # stable solver for small datasets
        ))
    ])
    return model
