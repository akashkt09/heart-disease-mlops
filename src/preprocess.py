# src/preprocess.py
"""Data loading, cleaning, and preprocessing pipeline for Heart Disease UCI dataset."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'thal']
BINARY_FEATURES = ['sex', 'fbs', 'exang', 'ca']


def load_data_from_uci():
    """Download the Heart Disease (Cleveland) dataset from UCI ML Repository."""
    from ucimlrepo import fetch_ucirepo

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize target and impute missing values in ca/thal columns."""
    df = df.copy()

    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])

    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer for feature preprocessing."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES)
        ]
    )


if __name__ == "__main__":
    df = load_data_from_uci()
    df = clean_data(df)
    df.to_csv('data/heart_disease_cleaned.csv', index=False)
    print(f"Cleaned dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
