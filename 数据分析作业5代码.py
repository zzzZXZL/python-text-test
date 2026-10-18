import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
USE_PCA_FOR_RBF = False  # Change to True to reduce RBF SVM training time.


def load_mnist_binary():
    """Load MNIST and retain only handwritten digits 0 and 1."""
    print("Loading MNIST from OpenML (first download may take a while)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(str)

    mask = np.isin(y, ["0", "1"])
    X = X[mask]
    y = y[mask].astype(int)

    print(f"Filtered samples: {len(y)}")
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index().rename(index={0: "digit 0", 1: "digit 1"}))
    return X, y


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    X, y = load_mnist_binary()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Linear SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, random_state=RANDOM_STATE, dual="auto", max_iter=10000),
        ),
        "RBF SVM": make_pipeline(
            StandardScaler(),
            *([PCA(n_components=0.95, random_state=RANDOM_STATE)] if USE_PCA_FOR_RBF else []),
            SVC(kernel="rbf", C=5.0, gamma="scale"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Training time (seconds)": train_seconds,
            }
        )

