import numpy as np
import pandas as pd


class SentimentProcessor:
    @staticmethod
    def _add_polarity_count(df: pd.DataFrame, col_suffix: str) -> pd.DataFrame:
        df = df.copy()
        polarities = ['POSITIVE', 'NEGATIVE', 'NONE']
        for polarity in polarities:
            columns = [col for col in df.columns if polarity in col]
            df[f'total_{polarity.lower()}_{col_suffix}'] = df[columns].sum(axis=1)
        return df

    @staticmethod
    def add_polarity_score(df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
        df = df.copy()
        df = SentimentProcessor._add_polarity_count(df, col_prefix)
        n_positive = df[f'total_positive_{col_prefix}']
        n_negative = df[f'total_negative_{col_prefix}']
        n_none = df[f'total_none_{col_prefix}']
        n_total = n_positive + n_negative + n_none
        df[f'{col_prefix}_score'] = (n_positive - n_negative) / n_total
        return df

    @staticmethod
    def add_pos_neg_ratio(df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
        df = df.copy()
        df = SentimentProcessor._add_polarity_count(df, col_prefix)
        n_positive = df[f'total_positive_{col_prefix}']
        n_negative = df[f'total_negative_{col_prefix}']
        offset = 0.5  # to avoid zero division
        df[f'{col_prefix}_pos_neg_ratio'] = np.log((n_positive + offset) / (n_negative + offset))
        return df
