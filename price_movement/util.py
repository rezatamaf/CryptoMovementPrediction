import datetime as dt
import glob
import logging
import json
import re

import pandas as pd
import pygsheets
from sklearn.model_selection import train_test_split
from pandas.api.types import is_object_dtype

from sentiment.constant import NLPDataConstants


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
    def parse_date(df: pd.DataFrame, date_col='date') -> pd.DataFrame:
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(by=date_col).set_index(date_col)

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
    def check_text_df_validity(df: pd.DataFrame):
        date_column_exist = 'date' in df.columns
        date_column_type_is_object = is_object_dtype(df.date)
        there_is_null_date = df.date.isnull().any()
        if date_column_exist and (date_column_type_is_object or there_is_null_date):
            raise ValueError("Date column contain NaN or mixed with other data, please check the csv file!")
        if not df[NLPDataConstants.COL_NAMES['ARTICLE']].is_unique:
            logging.warning("Found some duplicated article!")

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
    def dump_processing_output(model_output: dict, out_dir: str, coin_symbol='btc'):
        media_source = out_dir.split('/')[-1].lower()
        file_name = f'{coin_symbol}_{media_source}_sentiment_{model_output["date"]}.json'
        path = f'{out_dir}/{file_name}'
        with open(path, 'w') as f:
            json.dump(model_output, f)
        return model_output


class GSheetUpdater:
    def __init__(self, credential_path: str):
        self.credential_path = credential_path
        self.client = self._authorize()

    def _authorize(self):
        client = pygsheets.authorize(service_file=self.credential_path)
        return client

    def _get_worksheet(self, spreadsheet_name: str, worksheet_name: str):
        spreadsheet = self.client.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet_by_title(worksheet_name)
        return worksheet

    @staticmethod
    def _get_all_cell_values(worksheet: pygsheets.Worksheet, **kwargs):
        values = worksheet.get_all_values(returnas='matrix', **kwargs)
        return values

    def update(self, model_output: dict,
               spreadsheet_name: str,
               prediction_result_ws_name: str,
               tomorrow_prediction_ws_name: str):
        prediction_result_worksheet = self._get_worksheet(spreadsheet_name, prediction_result_ws_name)
        tomorrow_prediction_worksheet = self._get_worksheet(spreadsheet_name, tomorrow_prediction_ws_name)
        prediction_result_cells = self._get_all_cell_values(prediction_result_worksheet)
        tomorrow_prediction_cells = self._get_all_cell_values(tomorrow_prediction_worksheet)
        try:
            self.append_prediction_result(model_output, spreadsheet_name,
                                          prediction_result_ws_name, tomorrow_prediction_ws_name)
            self.update_tomorrow_prediction(model_output, spreadsheet_name, tomorrow_prediction_ws_name)
        except Exception as e:
            logging.error(e)
            logging.info("Rollback all updates")
            prediction_result_worksheet.update_values('A1', values=prediction_result_cells)
            tomorrow_prediction_worksheet.update_values('A1', values=tomorrow_prediction_cells)

    def append_prediction_history(self, model_output: dict, spreadsheet_name: str,
                                  worksheet_name: str):
        predicted_at = Utils.str_to_datetime(model_output['date'])
        predicted_for = predicted_at + dt.timedelta(days=1, hours=7)
        new_prediction_history_row = [predicted_for,
                                      model_output['tomorrow_price_up_prob'],
                                      model_output['tomorrow_prediction'],
                                      model_output['twitter_positive_sentiment'],
                                      model_output['twitter_negative_sentiment'],
                                      model_output['google_trends'],
                                      model_output['news_sentiment']
                                      ]
        worksheet = self._get_worksheet(spreadsheet_name, worksheet_name)
        cells = self._get_all_cell_values(worksheet, include_tailing_empty_rows=False, include_tailing_empty=False)
        last_row_idx = len(cells)
        worksheet.insert_rows(last_row_idx, number=1, values=new_prediction_history_row)
        added_row = worksheet.get_row(last_row_idx + 1, include_tailing_empty=False)
        if len(added_row) > 0:
            logging.info(f"Appending prediction history success on row #{last_row_idx + 1}!")

    def append_prediction_result(self, model_output: dict, spreadsheet_name: str,
                                 prediction_result_ws_name: str,
                                 tomorrow_prediction_ws_name: str):
        previous_day_date = Utils.str_to_datetime(model_output['date'])
        previous_day_date_with_hour = previous_day_date + dt.timedelta(hours=7)
        new_reference_price = model_output['reference_price']
        tomorrow_prediction_worksheet = self._get_worksheet(spreadsheet_name, tomorrow_prediction_ws_name)
        last_prediction_row = tomorrow_prediction_worksheet.get_row(2, include_tailing_empty=False)
        prediction_result_worksheet = self._get_worksheet(spreadsheet_name, prediction_result_ws_name)
        prediction_result_values = prediction_result_worksheet.get_all_values(include_tailing_empty_rows=False,
                                                                              include_tailing_empty=False,
                                                                              returnas='matrix')
        last_prediction_result_date = prediction_result_values[-1][0]
        last_prediction_result_datetime = Utils.str_to_datetime(last_prediction_result_date, fmt='%Y-%m-%d %H:%M:%S')
        new_prediction_result_datestr = Utils.datetime_to_str(previous_day_date_with_hour, fmt='%Y-%m-%d %H:%M:%S')
        if last_prediction_result_datetime == previous_day_date_with_hour:
            logging.warning(f"The prediction result for {last_prediction_result_date} already exist,"
                            f" no update will be done")
            return
        if len(last_prediction_row) > 0:
            last_reference_price = float(last_prediction_row[1].replace(',', '.'))
            last_prediction = last_prediction_row[2]
            price_diff = new_reference_price - last_reference_price
            up_prediction_correct = (price_diff > 0) and (last_prediction == 'Up')
            down_prediction_correct = (price_diff < 0) and (last_prediction == 'Down')
            prediction_correct = up_prediction_correct or down_prediction_correct
            prediction = 'Correct' if prediction_correct else 'Wrong'
        else:
            price_diff = ''
            prediction = ''
        new_prediction_result_row = [new_prediction_result_datestr,
                                     new_reference_price,
                                     previous_day_date.day,
                                     price_diff,
                                     prediction
                                     ]  # ordered as g-sheet's column order
        last_row_idx = len(prediction_result_values)
        prediction_result_worksheet.update_values(f'A{last_row_idx + 1}', values=[new_prediction_result_row])
        added_row = prediction_result_worksheet.get_row(last_row_idx + 1, include_tailing_empty=False)
        if len(added_row) > 0:
            logging.info(f"Appending prediction result success on row #{last_row_idx + 1}!")

    def update_tomorrow_prediction(self, model_output: dict, spreadsheet_name: str,
                                   worksheet_name: str):
        predicted_at = Utils.str_to_datetime(model_output['date']).date()
        predicted_for = predicted_at + dt.timedelta(1)
        tomorrow_prediction_worksheet = self._get_worksheet(spreadsheet_name, worksheet_name)
        tomorrow_prediction_worksheet_values = tomorrow_prediction_worksheet.get_all_values(
            include_tailing_empty_rows=False,
            include_tailing_empty=False,
            returnas='matrix')
        last_prediction_date = tomorrow_prediction_worksheet_values[-1][0]
        last_prediction_datetime = Utils.str_to_datetime(last_prediction_date, fmt='%d/%m/%Y')
        new_prediction_datestr = Utils.datetime_to_str(predicted_for, fmt='%d/%m/%Y')
        if last_prediction_datetime.date() == predicted_for:
            logging.warning(f"The prediction for {last_prediction_date} already exist,"
                            f" no update will be done")
            return
        new_row = [new_prediction_datestr,
                   model_output['reference_price'],
                   model_output['tomorrow_prediction'],
                   model_output['twitter_positive_sentiment'],
                   model_output['twitter_negative_sentiment'],
                   model_output['google_trends'],
                   ]
        worksheet = self._get_worksheet(spreadsheet_name, worksheet_name)
        worksheet.update_row(2, values=new_row)
        logging.info(f"Updating tomorrow Prediction success!")
