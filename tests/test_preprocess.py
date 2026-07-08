"""Unit tests for data preprocessing logic."""

import pandas as pd
import pytest

from src.preprocess import clean_data, build_preprocessor


@pytest.fixture
def raw_sample_df():
    """A small synthetic sample mimicking the raw UCI Heart Disease format."""
    return pd.DataFrame({
        'age': [63, 67, 41],
        'sex': [1, 1, 0],
        'cp': [1, 4, 2],
        'trestbps': [145, 160, 130],
        'chol': [233, 286, 204],
        'fbs': [1, 0, 0],
        'restecg': [2, 2, 2],
        'thalach': [150, 108, 172],
        'exang': [0, 1, 0],
        'oldpeak': [2.3, 1.5, 1.4],
        'slope': [3, 2, 1],
        'ca': [0.0, 3.0, None],
        'thal': [6.0, 3.0, None],
        'num': [0, 2, 0]
    })


def test_clean_data_binarizes_target(raw_sample_df):
    df_clean = clean_data(raw_sample_df)
    assert 'target' in df_clean.columns
    assert 'num' not in df_clean.columns
    assert set(df_clean['target'].unique()).issubset({0, 1})


def test_clean_data_target_values_correct(raw_sample_df):
    df_clean = clean_data(raw_sample_df)
    assert df_clean['target'].tolist() == [0, 1, 0]


def test_clean_data_imputes_missing_values(raw_sample_df):
    df_clean = clean_data(raw_sample_df)
    assert df_clean['ca'].isnull().sum() == 0
    assert df_clean['thal'].isnull().sum() == 0


def test_clean_data_does_not_mutate_original(raw_sample_df):
    original_columns = raw_sample_df.columns.tolist()
    _ = clean_data(raw_sample_df)
    assert raw_sample_df.columns.tolist() == original_columns
    assert 'num' in raw_sample_df.columns


def test_build_preprocessor_returns_column_transformer():
    from sklearn.compose import ColumnTransformer
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 3
