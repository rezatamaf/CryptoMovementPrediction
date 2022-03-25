import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureProcessor:
    @staticmethod
    def _add_polarity_count(df: pd.DataFrame, col_suffix: str) -> pd.DataFrame:
        df = df.copy()
        polarities = ['POSITIVE', 'NEGATIVE', 'NONE']
        for polarity in polarities:
            columns = [col for col in df.columns if polarity in col]
            df[f'total_{polarity.lower()}_{col_suffix}'] = df[columns].sum(axis=1)
        return df

    @staticmethod
    def get_log_ratio(numerator, denumerator, offset=0.5) -> float:
        return np.log((numerator + offset) / (denumerator.shift(1) + offset))

    @staticmethod
    def add_polarity_score(df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
        df = df.copy()
        df = FeatureProcessor._add_polarity_count(df, col_prefix)
        n_positive = df[f'total_positive_{col_prefix}']
        n_negative = df[f'total_negative_{col_prefix}']
        n_none = df[f'total_none_{col_prefix}']
        n_total = n_positive + n_negative + n_none
        df[f'{col_prefix}_score'] = (n_positive - n_negative) / n_total
        return df

    @staticmethod
    def add_pos_neg_ratio(df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
        df = df.copy()
        df = FeatureProcessor._add_polarity_count(df, col_prefix)
        n_positive = df[f'total_positive_{col_prefix}']
        n_negative = df[f'total_negative_{col_prefix}']
        offset = 0.5  # to avoid zero division
        df[f'{col_prefix}_pos_neg_ratio'] = FeatureProcessor.get_log_ratio(n_positive, n_negative, offset)
        return df

    @staticmethod
    def get_past_sequence(df: pd.DataFrame, col: str, n_days=3) -> int:
        sequence = ''
        for i in range(n_days):
            sequence += df[col].shift(i + 1).astype(str)
        le = LabelEncoder()
        encoded_labels = le.fit_transform(sequence)
        return encoded_labels


def select_features(clf, X, y, n_features=10):
    clf.fit(X, y)
    sorted_idx = clf.feature_importances_.argsort()[::-1][:n_features]
    selected_cols = X.columns[sorted_idx].values.tolist()
    return selected_cols
