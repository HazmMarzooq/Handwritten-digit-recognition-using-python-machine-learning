"""Train, save, load and use the MLP digit classifier (scikit-learn)."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

MODEL_PATH = Path(__file__).parent / "digits_mlp.joblib"
MAX_PIXEL = 16.0  # load_digits() pixel values range from 0 to 16


def train_model() -> dict:
    """Train an MLPClassifier on sklearn's built-in digits dataset."""
    digits = load_digits()
    X = digits.data / MAX_PIXEL          # (1797, 64) scaled to 0..1
    y = digits.target                    # labels 0..9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=500,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    bundle = {
        "model": clf,
        "train_accuracy": accuracy_score(y_train, clf.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, clf.predict(X_test)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train() -> dict:
    """Load the saved model, or train it on first run."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass  # corrupted / incompatible version -> retrain
    return train_model()


def predict_digit(model: MLPClassifier, features: np.ndarray):
    """features: 64 pixel values on the 0..16 scale.
    Returns (digit, confidence 0..1, probabilities for digits 0..9)."""
    x = np.asarray(features, dtype=float).reshape(1, 64) / MAX_PIXEL
    probs = model.predict_proba(x)[0]
    digit = int(model.classes_[np.argmax(probs)])
    return digit, float(probs.max()), probs


if __name__ == "__main__":
    b = train_model()
    print(f"Train accuracy: {b['train_accuracy']:.4f}")
    print(f"Test accuracy:  {b['test_accuracy']:.4f}")
    print(f"Saved to {MODEL_PATH}")
