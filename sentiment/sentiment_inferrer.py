import datetime as dt
import glob
import logging
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.api.types import is_object_dtype

from sentiment import bert_preprocessor, text_preprocessor
from sentiment.constant import NLPDataConstants, ModelConstants
from sentiment.model_loader import ModelLoader
from price_movement.util import Utils


class Inferrer:
    def run(self, model: ModelLoader, input_dir: str, output_dir: str, date: str = None, sample_size=1000):
        media_source = input_dir.split('/')[-1].lower()
        logging.info(f"START PROCESSING {media_source.upper()} DATA")
        try:
            logging.info(f'Loading {date} data ...')
            df = self._load_df(input_dir, date)
        except IndexError as error:
            logging.warning(f'{error}\n')
            return
        except Exception as error:
            logging.exception(error)
            raise

        # combine title and article as content column
        if 'title' in df.columns:
            df['content'] = df['title'] + ' ' + df['article']
        else:
            df['content'] = df['article']

        # sample data
        if len(df) > sample_size:
            sample_df = df.sample(sample_size)
            logging.info(f'{len(sample_df)} rows sampled from {len(df)} rows'
                         f' ({np.round(len(sample_df) / len(df) * 100, 2)}%)')
        else:
            sample_df = df.copy()

        # preprocess text
        sample_df = self.preprocess_content(sample_df, content_col='content')
        # transform sample data into tf dataset
        sample_dataset = bert_preprocessor.create_dataset(sample_df,
                                                          text_col='content',
                                                          tokenizer=model.tokenizer).batch(32)

        # filter relevant content
        logging.info("Checking relevance ...")
        filtered_df = self.filter_relevance(model, sample_dataset, sample_df)
        logging.info(f'{len(filtered_df)}/{len(sample_df)} rows are detected as relevant')

        # stop if no relevant content is found
        if len(filtered_df) == 0:
            logging.info("DONE\n")
            return

        # transform filtered df to tabsa-formatted dataset
        tabsa_dataset = self.process_content_to_tabsa_format(model, filtered_df.content)

        # infer sentiment
        logging.info("Inferring sentiment ...")
        predicted_sentiment = self.infer_sentiment(model, tabsa_dataset)

        logging.info("Processing output result ...")
        # generate json output
        output = self.generate_output(predicted_sentiment, df)
        logging.info("Dump result to gdrive ...")
        self.dump_output(output, output_dir)
        logging.info("DONE\n")
        return output

    @staticmethod
    def _load_df(data_dir: str, date: str) -> pd.DataFrame:
        if date is None:
            date = Utils.datetime_to_str(dt.datetime.today() - dt.timedelta(1))
        files = glob.glob(f'{data_dir}/*{date}.csv')
        if len(files) == 0:
            raise IndexError("No data found in selected date!")
        elif len(files) > 1:
            raise ValueError("Duplicate data in selected date are detected!")
        file = files[0]
        column_names = list(NLPDataConstants.COL_NAMES.values())
        column_dtypes = NLPDataConstants.COL_DTYPES.copy()
        try:
            df = pd.read_csv(file, usecols=column_names, dtype=column_dtypes, parse_dates=['date'])
        except ValueError as e:
            if 'usecols' in str(e).lower():
                column_names.remove(NLPDataConstants.COL_NAMES['TITLE'])
                column_dtypes.pop(NLPDataConstants.COL_NAMES['TITLE'])
                df = pd.read_csv(file, usecols=column_names, dtype=column_dtypes, parse_dates=['date'])
            else:
                raise
        if (is_object_dtype(df.date)) or (df.date.isnull().any()):
            raise ValueError("Date column contain NaN or mixed with other data, please check the csv file!")
        if not df[NLPDataConstants.COL_NAMES['ARTICLE']].is_unique:
            logging.warning("Found some duplicated article!")
        return df

    @staticmethod
    def preprocess_content(df: pd.DataFrame, content_col: str) -> pd.DataFrame:
        df[content_col] = df.content.apply(lambda x: text_preprocessor.normalize_tweet(x))
        return df

    @staticmethod
    def filter_relevance(model: ModelLoader, tf_dataset: tf.data.Dataset, source_df: pd.DataFrame) -> pd.DataFrame:
        pred_prob = model.relevance_model.predict(tf_dataset)
        is_relevance = np.argmax(pred_prob, axis=1) == 1
        filtered_df = source_df[is_relevance].reset_index(drop=True)
        return filtered_df

    @staticmethod
    def process_content_to_tabsa_format(model: ModelLoader, contents: pd.Series,
                                        coin_symbol='btc', coin_name='bitcoin') -> tf.data.Dataset:
        # duplicate contents for every aspect category
        duplicated_contents = []
        n_aspect_categories = len(ModelConstants.ASPECT_CATEGORY)
        for content in contents:
            duplicated_contents += [content] * n_aspect_categories
        # create pseudo sentence as mentioned in tabsa paper
        pseudo_sentence = [f"{coin_symbol} - {coin_name} - {category.lower()}"
                           for category in ModelConstants.ASPECT_CATEGORY]
        # duplicate pseudo sentence for every content
        duplicated_pseudo_sentence = pseudo_sentence * len(contents)
        df = pd.DataFrame({'content': duplicated_contents,
                           'pseudo_sentence': duplicated_pseudo_sentence})
        # transform dataframe to tf dataset
        dataset = bert_preprocessor.create_dataset(df, label_col=None, text_col='content',
                                                   text_pair_col='pseudo_sentence', tokenizer=model.tokenizer)
        return dataset.batch(32)

    @staticmethod
    def infer_sentiment(model: ModelLoader, dataset: tf.data.Dataset) -> pd.Series:
        polarity_prob = model.tabsa_model.predict(dataset)
        polarity_class = pd.Series(np.argmax(polarity_prob, axis=1)).map(ModelConstants.ID2POLARITY)
        return polarity_class

    @staticmethod
    def _count_sentiment_distribution(predicted_sentiment: pd.Series) -> dict:
        # make list of aspect-polarity pairs
        n_contents = len(predicted_sentiment) // len(ModelConstants.ASPECT_CATEGORY)
        aspect_polarity_tuple = list(zip(ModelConstants.ASPECT_CATEGORY * n_contents, predicted_sentiment))
        # convert to dataframe for easier processing
        category_polarity_df = pd.DataFrame(aspect_polarity_tuple, columns=['category', 'polarity'])
        # count occurrences of aspect-polarity pairs
        sentiment_distribution = category_polarity_df.groupby(['category', 'polarity']).size().to_dict()
        # populate dict with counted aspect-polarity pairs occurrences
        sentiment_dict = {category: {} for category in ModelConstants.ASPECT_CATEGORY}
        for i in sentiment_distribution:
            sentiment_dict[i[0]][i[1]] = sentiment_dict.get(i[1], sentiment_distribution[i])
        return sentiment_dict

    @staticmethod
    def generate_output(predicted_sentiment: pd.Series, unsampled_df: pd.DataFrame) -> dict:
        sentiment_dist = Inferrer._count_sentiment_distribution(predicted_sentiment)
        inference_date = Utils.datetime_to_str(unsampled_df.date.dt.date.values[0])
        output = {'date': inference_date,
                  'total_docs': len(unsampled_df),
                  'sentiment': sentiment_dist}
        return output

    @staticmethod
    def dump_output(model_output: dict, out_dir: str, coin_symbol='btc'):
        media_source = out_dir.split('/')[-1].lower()
        file_name = f'{coin_symbol}_{media_source}_sentiment_{model_output["date"]}.json'
        path = f'{out_dir}/{file_name}'
        with open(path, 'w') as f:
            json.dump(model_output, f)
        return model_output
