import datetime as dt
import numpy as np
import pandas as pd

from price_movement.util import Utils
from price_movement.feature_processor import SentimentProcessor


class DataLoader:
    def __init__(self,
                 twitter_sentiment_dir: str,
                 cnn_sentiment_dir: str,
                 btcnews_sentiment_dir: str,
                 gtrend_dir: str,
                 binance_price_dir: str
                 ):
        self.twitter_sentiment_dir = twitter_sentiment_dir
        self.cnn_sentiment_dir = cnn_sentiment_dir
        self.btcnews_sentiment_dir = btcnews_sentiment_dir
        self.gtrend_dir = gtrend_dir
        self.binance_price_dir = binance_price_dir
        self.twitter_positive_sentiment = 0
        self.twitter_negative_sentiment = 0
        self.news_sentiment = 0
        self.google_trends = 0
        self.test_date = ''
        self.reference_price = 0

    def run(self, training_period=30, today_reference: str = None):
        twitter_df = self._load_sentiment(self.twitter_sentiment_dir, training_period, today_reference)
        cnn_df = self._load_sentiment(self.cnn_sentiment_dir, training_period, today_reference)
        gtrend_df = self._load_gtrend(self.gtrend_dir, training_period, today_reference)
        btcnews_df = self._load_sentiment(self.btcnews_sentiment_dir, training_period, today_reference)
        price_df = self._load_price(self.binance_price_dir, training_period, today_reference)

        # concat all data to one dataframe
        dfs = [twitter_df, cnn_df, btcnews_df, gtrend_df, price_df]
        data_df = pd.concat(dfs, join='outer', axis=1).fillna(0)[1:]  # drop 1st row after lagged preprocessing
        complete_date_idx = Utils.get_relevant_dates(training_period, today_reference)[1:]
        data_df = data_df.reindex(complete_date_idx).fillna(0)  # add missing date if any

        # extract some info from test date
        self.test_date = np.datetime_as_string(data_df.iloc[[-1]].index.values[0], unit='D')
        self.reference_price = data_df.iloc[[-1]]['close_price'].values[0]
        self.twitter_negative_sentiment = data_df.iloc[[-1]]['total_negative_twitter'].values[0]
        self.twitter_positive_sentiment = data_df.iloc[[-1]]['total_positive_twitter'].values[0]
        self.news_sentiment = data_df.iloc[[-1]]['bitcoin_news_score'].values[0]
        self.google_trends = data_df.iloc[[-1]]['trends'].values[0]

        # drop non-features columns for sentiment dfs
        non_features = '(^(?!total))(^(?!sentiment))'
        data_df = data_df.filter(regex=non_features)
        return data_df

    @staticmethod
    def _load_sentiment(data_dir: str, training_period: int, today_reference: str) -> pd.DataFrame:
        sentiment_df = Utils.load_relevant_jsons(data_dir, training_period, today_reference)
        col_prefix = data_dir.split('/')[-1].lower()  # use dir_name as prefix
        sentiment_df = SentimentProcessor.add_polarity_score(sentiment_df, col_prefix)
        sentiment_df = SentimentProcessor.add_pos_neg_ratio(sentiment_df, col_prefix)
        return sentiment_df

    @staticmethod
    def _load_price(data_dir: str, training_period: int, today_reference: str) -> pd.DataFrame:
        price_df = Utils.load_relevant_jsons(data_dir, training_period, today_reference)
        close_price = price_df['close_price']
        tomorrow_close_price = close_price.shift(-1)
        price_df['price_diff'] = np.log((close_price + 0.5) / (close_price.shift(1) + 0.5))
        price_df['is_price_up'] = (tomorrow_close_price - close_price) > 0
        price_df.loc[tomorrow_close_price.isnull(), 'is_price_up'] = np.NaN
        return price_df

    @staticmethod
    def _load_gtrend(data_dir: str, training_period: int, today_reference: str) -> pd.DataFrame:
        gtrend_df = Utils.load_relevant_jsons(data_dir, training_period, today_reference)

        adj_index = [i + dt.timedelta(3) for i in gtrend_df.index]
        gtrend_df.index = adj_index
        gtrend_df.index.name = 'date'
        return gtrend_df
