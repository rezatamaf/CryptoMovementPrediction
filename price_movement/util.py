import datetime as dt
import glob
import logging
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split


class Utils:
    @staticmethod
    def count_usable_data(data_dir: str, today_reference: str, offset=2):
        today = Utils.get_reference_day_in_date(today_reference)
        files = glob.glob(f'{data_dir}/*.json')
        usable_files = [f for f in files if Utils.extract_file_date(f) <= today]
        total_usable_data = len(usable_files) - offset
        return total_usable_data

    @staticmethod
    def extract_file_date(filename: str):
        file_date_str = Utils.extract_date_from_text(filename)
        file_date = Utils.str_to_datetime(file_date_str).date()
        return file_date

    @staticmethod
    def get_relevant_dates(training_period: int, today_reference: str) -> pd.DatetimeIndex:
        today = Utils.get_reference_day_in_date(today_reference)
        yesterday = today - dt.timedelta(1)
        day_offset = 2  # 1 day for lagged processing, 1 day for predicting
        start_date = today - dt.timedelta(training_period + day_offset)
        end_date = yesterday
        relevant_dates = pd.date_range(start_date, end_date, freq='d')
        return relevant_dates

    @staticmethod
    def get_reference_day_in_date(reference_day: str = None):
        if reference_day is None:
            day_in_date = dt.datetime.today().date()
        else:
            day_in_date = Utils.str_to_datetime(reference_day).date()
        return day_in_date

    @staticmethod
    def get_relevant_files(data_dir: str, training_period: int, today_reference: str) -> list:
        files = glob.glob(f'{data_dir}/*.json')
        relevant_dates = Utils.get_relevant_dates(training_period, today_reference)
        relevant_dates_in_str = [Utils.datetime_to_str(i) for i in relevant_dates]
        relevant_files = [i for i in files if Utils.extract_date_from_text(i) in relevant_dates_in_str]
        # check duplicate files
        if len(relevant_files) != len(set(relevant_files)):
            raise ValueError('Found multiple files with same date!')
        return relevant_files

    @staticmethod
    def get_sample_files(data_dir: str) -> list:
        files = glob.glob(f'{data_dir}/*.json')
        sample_files = [files[0]]
        return sample_files

    @staticmethod
    def datetime_to_str(date_time, fmt='%Y-%m-%d') -> str:
        return date_time.strftime(fmt)

    @staticmethod
    def str_to_datetime(text: str, fmt='%Y-%m-%d'):
        return dt.datetime.strptime(text, fmt)

    @staticmethod
    def extract_date_from_text(text: str, pattern=r'\d{4}-\d{2}-\d{2}'):
        return re.search(pattern, text).group()

    @staticmethod
    def shift_date_str(date: str, days_lag: int):
        original_date_str = Utils.str_to_datetime(date) + dt.timedelta(days_lag)
        return Utils.datetime_to_str(original_date_str)

    @staticmethod
    def parse_date(df: pd.DataFrame, date_col='date') -> pd.DataFrame:
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(by=date_col).set_index(date_col)

    @staticmethod
    def adjust_date_index(df: pd.DataFrame, days_lag: int) -> pd.DataFrame:
        df = df.copy()
        adjusted_index = [i + dt.timedelta(days_lag) for i in df.index]
        df.index = adjusted_index
        return df

    @staticmethod
    def load_relevant_jsons(data_dir: str, training_period: int, today_reference: str) -> pd.DataFrame:
        media_source = data_dir.split('/')[-1].lower()
        try:
            files = Utils.get_relevant_files(data_dir, training_period, today_reference)
        except ValueError as error:
            logging.error(error)
            raise
        if len(files) == 0:
            logging.warning(f"No {media_source} data found on the relevant dates!")
            files = Utils.get_sample_files(data_dir)
        dfs = []
        for i in files:
            with open(i) as f:
                json_data = pd.json_normalize(json.loads(f.read()))
            dfs.append(json_data)
        data_df = pd.concat(dfs, sort=False).fillna(0)
        return Utils.parse_date(data_df)

    @staticmethod
    def get_csv_by_date(data_dir: str, date: str):
        if date is None:
            date = Utils.datetime_to_str(dt.datetime.today() - dt.timedelta(1))
        files = glob.glob(f'{data_dir}/*{date}.csv')
        if len(files) == 0:
            raise IndexError("No data found in selected date!")
        elif len(files) > 1:
            raise ValueError("Duplicate data in selected date are detected!")
        file = files[0]
        return file

    @staticmethod
    def split_data(df: pd.DataFrame) -> tuple:
        X = df.drop(['is_price_up', 'close_price'], axis=1)
        y = df['is_price_up']
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=1, shuffle=False)
        return X_train, X_test, y_train

    @staticmethod
    def dump_prediction_output(model_output: dict, out_dir: str):
        dir_name = out_dir
        file_name = f'prediction_{model_output["date"]}.json'
        path = f'{dir_name}/{file_name}'
        with open(path, 'w') as f:
            json.dump(model_output, f)

    @staticmethod
    def dump_processing_output(model_output: dict, out_dir: str, coin_symbol='btc', suffix=None):
        media_source = out_dir.split('/')[-1].lower()
        if suffix is not None:
            file_name = f'{coin_symbol}_{media_source}_{suffix}_{model_output["date"]}.json'
        else:
            file_name = f'{coin_symbol}_{media_source}_{model_output["date"]}.json'
        path = f'{out_dir}/{file_name}'
        with open(path, 'w') as f:
            json.dump(model_output, f)
        return model_output

