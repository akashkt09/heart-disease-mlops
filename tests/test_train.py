"""Unit tests for model training logic."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor
from src.train import evaluate_model


@pytest.fixture
def synthetic_dataset():
    """A larger synthetic dataset with the right schema for training a real pipeline."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'age': np.random.randint(30, 80, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(1, 5, n),
        'trestbps': np.random.randint(100, 180, n),
        'chol': np.random.randint(150, 350, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(90, 200, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.random.uniform(0, 5, n),
        'slope': np.random.randint(1, 4, n),
        'ca': np.random.randint(0, 4, n).astype(float),
        'thal': np.random.choice([3.0, 6.0, 7.0], n),
    })
    target = np.random.randint(0, 2, n)
    return df, target


def test_pipeline_fits_and_predicts(synthetic_dataset):
    X, y = synthetic_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_evaluate_model_returns_expected_keys(synthetic_dataset):
    X, y = synthetic_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'roc_auc'}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
