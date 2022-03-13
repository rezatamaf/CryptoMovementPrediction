import datetime as dt
import numpy as np

from price_movement.data_loader import DataLoader


def generate_output(data_loader: DataLoader, predict_proba: np.array, model_threshold: float) -> dict:
    price_up_prob = predict_proba[0, 1]
    prediction = 'Up' if price_up_prob > model_threshold else 'Down'
    utc_date = data_loader.test_date  # date(UTC)
    wib_datetime = dt.datetime.strptime(data_loader.test_date, "%Y-%m-%d") + dt.timedelta(1)
    wib_date = wib_datetime.strftime('%Y-%m-%d')
    output = {'utc_date': utc_date,
              'date': wib_date,
              'reference_price': float(data_loader.reference_price),
              'tomorrow_price_up_prob': round(float(price_up_prob), 2),
              'tomorrow_prediction': prediction,
              'twitter_positive_sentiment': int(data_loader.twitter_positive_sentiment),
              'twitter_negative_sentiment': int(data_loader.twitter_negative_sentiment),
              'news_sentiment': round(float(data_loader.news_sentiment), 2),
              'google_trends': int(data_loader.google_trends)
              }
    return output
