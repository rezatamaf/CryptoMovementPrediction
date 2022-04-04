import datetime as dt
import logging

import pandas as pd
import pygsheets

from price_movement.data_loader import DataLoader
from price_movement.util import Utils


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

    def update(self, model_output: dict, data_loader: DataLoader,
               spreadsheet_name: str,
               prediction_result_ws_name: str,
               tomorrow_prediction_ws_name: str,
               historical_search_trend_ws_name: str,
               historical_sentiment_ws_name: str):
        prediction_result_worksheet = self._get_worksheet(spreadsheet_name, prediction_result_ws_name)
        tomorrow_prediction_worksheet = self._get_worksheet(spreadsheet_name, tomorrow_prediction_ws_name)
        historical_search_trend_worksheet = self._get_worksheet(spreadsheet_name, historical_search_trend_ws_name)
        historical_sentiment_worksheet = self._get_worksheet(spreadsheet_name, historical_sentiment_ws_name)
        prediction_result_cells = self._get_all_cell_values(prediction_result_worksheet)
        tomorrow_prediction_cells = self._get_all_cell_values(tomorrow_prediction_worksheet)
        historical_search_trend_cells = self._get_all_cell_values(historical_search_trend_worksheet)
        historical_sentiment_cells = self._get_all_cell_values(historical_sentiment_worksheet)
        try:
            self.append_prediction_result(model_output, spreadsheet_name,
                                          prediction_result_ws_name, tomorrow_prediction_ws_name)
            self.update_tomorrow_prediction(model_output, spreadsheet_name, tomorrow_prediction_ws_name)
            self.update_historical_search_trend(data_loader, spreadsheet_name, historical_search_trend_ws_name)
            self.update_historical_sentiment(data_loader, spreadsheet_name, historical_sentiment_ws_name)
        except Exception as e:
            logging.error(e)
            logging.info("Rollback all updates")
            prediction_result_worksheet.update_values('A1', values=prediction_result_cells)
            tomorrow_prediction_worksheet.update_values('A1', values=tomorrow_prediction_cells)
            historical_search_trend_worksheet.update_values('A1', values=historical_search_trend_cells)
            historical_sentiment_worksheet.update_values('A1', values=historical_sentiment_cells)

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

    def _update_historical_metrics(self, metrics: pd.DataFrame, spreadsheet_name: str, worksheet_name: str):
        metric_name = metrics.name
        replacement_cells = metrics.values.tolist()
        worksheet = self._get_worksheet(spreadsheet_name, worksheet_name)
        worksheet.update_values('A2', values=replacement_cells)
        logging.info(f"Updating historical {metric_name} success!")

    def update_historical_sentiment(self, data_loader: DataLoader,
                                    spreadsheet_name: str, worksheet_name: str, n_days=7):
        n_days_adjustment = 2
        date_reference = Utils.shift_date_str(data_loader.test_date, days_lag=1)
        sentiment_df = data_loader.load_sentiment(data_loader.twitter_sentiment_dir,
                                                  n_days - n_days_adjustment, date_reference)
        sentiment_df = Utils.adjust_date_index(sentiment_df, days_lag=1)
        sentiment_df = sentiment_df[['total_positive_twitter', 'total_negative_twitter']].reset_index()
        sentiment_df['date'] = sentiment_df['date'].dt.strftime('%Y-%m-%d')
        sentiment_df.name = 'Twitter Sentiment'
        self._update_historical_metrics(sentiment_df, spreadsheet_name, worksheet_name)

    def update_historical_search_trend(self, data_loader: DataLoader, spreadsheet_name: str,
                                       worksheet_name: str, n_days=7):
        date_reference = Utils.shift_date_str(data_loader.test_date, days_lag=1)
        trend_df = data_loader.load_gtrend(data_loader.gtrend_dir, n_days, date_reference, adjust_index=False)
        trend_df = Utils.adjust_date_index(trend_df, days_lag=1)
        trend_df = trend_df.reset_index()
        trend_df['date'] = trend_df['date'].dt.strftime('%Y-%m-%d')
        trend_df['trends'] = trend_df['trends'].apply(lambda x: round(x, 2))
        trend_df.name = 'Search Trend'
        self._update_historical_metrics(trend_df, spreadsheet_name, worksheet_name)
