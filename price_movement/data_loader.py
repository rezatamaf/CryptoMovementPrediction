import json
import requests
import datetime as dt
import numpy as np
import pandas as pd
from iso8601 import ParseError, parse_date

from price_movement.util import Utils
from price_movement.feature_processor import SentimentProcessor


class GlassnodeClient:

    def __init__(self):
        self._api_key = ''

    @property
    def api_key(self):
        return self._api_key

    def set_api_key(self, value):
        self._api_key = value

    def get(self, url, a='BTC', i='24h', c='native', s=None, u=None):
        p = dict()
        p['a'] = a
        p['i'] = i
        p['c'] = c

        if s is not None:
            try:
                p['s'] = parse_date(s).strftime('%s')
            except ParseError:
                p['s'] = s

        if u is not None:
            try:
                p['u'] = parse_date(u).strftime('%s')
            except ParseError:
                p['u'] = s

        p['api_key'] = self.api_key

        r = requests.get(url, params=p)

        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            print(r.text)

        try:
            df = pd.DataFrame(json.loads(r.text))
            df = df.set_index('t')
            df.index = pd.to_datetime(df.index, unit='s')
            df = df.sort_index()
            s = df.v
            s.name = '_'.join(url.split('/')[-2:])
            return s
        except Exception as e:
            print(e)


class FundamentalMetricLoader:
    def __init__(self, reference_date_idx: pd.Index, api_key: str,
                 base_url='https://api.glassnode.com/v1/metrics'):
        self.reference_date_idx = reference_date_idx
        self.base_url = base_url
        self.client = self._authorize(api_key)

    @staticmethod
    def _authorize(api_key: str):
        client = GlassnodeClient()
        client.set_api_key(api_key)
        return client

    def get_metric(self, end_point):
        since_date, until_date = self._get_date_range()
        return self.client.get(f'{self.base_url}/{end_point}', s=since_date, u=until_date).astype('float64')

    def _get_date_range(self):
        min_date = Utils.datetime_to_str(self.reference_date_idx.min())
        max_date = Utils.datetime_to_str(self.reference_date_idx.max() + dt.timedelta(1))  # api use exclusive boundary
        return min_date, max_date


class DataLoader:
    def __init__(self,
                 twitter_sentiment_dir: str,
                 cnn_sentiment_dir: str,
                 btcnews_sentiment_dir: str,
                 gtrend_dir: str,
                 binance_price_dir: str,
                 glassnode_api_path: str,
                 influencer_raw_dir: str,
                 influencer_sentiment_dir: str
                 ):
        self.twitter_sentiment_dir = twitter_sentiment_dir
        self.cnn_sentiment_dir = cnn_sentiment_dir
        self.btcnews_sentiment_dir = btcnews_sentiment_dir
        self.gtrend_dir = gtrend_dir
        self.binance_price_dir = binance_price_dir
        self.glassnode_api_path = glassnode_api_path
        self.influencer_raw_dir = influencer_raw_dir
        self.influencer_sentiment_dir = influencer_sentiment_dir
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
        fundamental_df = self._load_fundamental(self.glassnode_api_path, twitter_df.index)
        influencer_df = self._load_influencer(self.influencer_raw_dir, self.influencer_sentiment_dir,
                                              training_period, today_reference)
        price_df = self._load_price(self.binance_price_dir, training_period, today_reference)

        # concat all data to one dataframe
        dfs = [twitter_df, cnn_df, btcnews_df, gtrend_df, fundamental_df, influencer_df, price_df]
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

    @staticmethod
    def _load_fundamental(api_path: str, reference_date_idx: pd.Index):
        with open(api_path) as json_file:
            loaded_json = json.load(json_file)
        api_key = loaded_json['api_key']
        metric_loader = FundamentalMetricLoader(reference_date_idx, api_key)
        end_points = ['indicators/sopr', 'transactions/rate', 'mining/hash_rate_mean', 'addresses/active_count',
                      'blockchain/utxo_created_value_sum', 'blockchain/utxo_spent_value_sum']
        df = pd.DataFrame({i.split("/")[1]: metric_loader.get_metric(i) for i in end_points})
        df.index.names = ['date']
        return df

    @staticmethod
    def _load_influencer(raw_dir: str, sentiment_dir: str, training_period: int, today_reference: str):
        influencer_sentiment_df = DataLoader._load_sentiment(sentiment_dir, training_period, today_reference)
        return influencer_sentiment_df
