# src/train.py
"""Train, tune, and log heart disease classification models with MLflow."""

import pickle
import os

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.preprocess import load_data_from_uci, clean_data, build_preprocessor


def get_train_test_split(random_state=42):
    df = load_data_from_uci()
    df = clean_data(df)

    X = df.drop(columns=['target'])
    y = df['target']

    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train):
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }


def main():
    mlflow.set_experiment("heart-disease-classification")

    X_train, X_test, y_train, y_test = get_train_test_split()

    with mlflow.start_run(run_name="logistic_regression_tuned_FINAL"):
        grid_search = train_logistic_regression(X_train, y_train)
        best_model = grid_search.best_estimator_

        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_roc_auc", grid_search.best_score_)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(best_model, "model", serialization_format="pickle")
        mlflow.set_tag("final_model", "true")

        os.makedirs('model', exist_ok=True)
        with open('model/heart_disease_pipeline.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        print("Training complete. Metrics:", metrics)

    return best_model, metrics


if __name__ == "__main__":
    main()
