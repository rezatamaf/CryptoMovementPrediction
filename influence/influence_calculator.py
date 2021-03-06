import logging
import pandas as pd

from price_movement.util import Utils


class InfluenceCalculator:
    @staticmethod
    def run(influencer_list_dir: str, influencer_tweet_dir: str, output_dir: str, date: str = None):
        logging.info(f"START PROCESSING {date} INFLUENCER DATA")
        influencer_list_col_name = ['twitter_screen_name', 'influencer_rank_average']
        influencer_list_col_dtype = {'twitter_screen_name': str, 'influencer_rank_average': float}
        logging.info(f"Loading influencer list ...")
        influencer_list_df = InfluenceCalculator._load_df(influencer_list_dir, date,
                                                          influencer_list_col_name,
                                                          influencer_list_col_dtype,
                                                          alternative_date='2022-03-21')
        influencer_tweet_col_name = ['username_source', 'tweet_id']
        influencer_tweet_col_dtype = {'username_source': str, 'tweet_id': str}
        logging.info(f"Loading influencer tweets ...")
        influencer_tweet_df = InfluenceCalculator._load_df(influencer_tweet_dir, date,
                                                           influencer_tweet_col_name, influencer_tweet_col_dtype)
        influencer_df = pd.merge(influencer_list_df, influencer_tweet_df,
                                 left_on='twitter_screen_name', right_on='username_source')
        logging.info(f"Calculating {date} total influence score ...")
        total_influence_score = influencer_df['influencer_rank_average'].sum()
        output = InfluenceCalculator._generate_output(total_influence_score, date)
        logging.info("Dumping result to gdrive ...")
        Utils.dump_processing_output(output, output_dir, suffix=None)
        logging.info("DONE\n")
        return output

    @staticmethod
    def _load_df(data_dir: str, date: str, column_names: list,
                 column_dtypes: dict, alternative_date=None) -> pd.DataFrame:
        try:
            file = Utils.get_csv_by_date(data_dir, date)
        except IndexError as e:
            logging.warning(e)
            if alternative_date is not None:
                logging.info(f"Trying alternative date {alternative_date}")
                file = Utils.get_csv_by_date(data_dir, alternative_date)
            else:
                raise
        except Exception:
            raise
        df = pd.read_csv(file, usecols=column_names, dtype=column_dtypes)
        return df

    @staticmethod
    def _generate_output(total_influence_score: float, date: str) -> dict:
        output = {'date': date,
                  'influence_score': total_influence_score}
        return output
