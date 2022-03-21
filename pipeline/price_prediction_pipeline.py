import logging

from price_movement.data_loader import DataLoader
from price_movement.util import Utils, GSheetUpdater
from price_movement.price_classifier import Model
from price_movement.output_generator import generate_output
from price_movement.feature_processor import select_features
from price_movement.model_tuner import fine_tune_model

# Define Constants
DATA_DIR = '/content/drive/MyDrive/CryptoDataset'
TWITTER_SENTIMENT_DIR = f'{DATA_DIR}/sentiment/twitter'
CNN_SENTIMENT_DIR = f'{DATA_DIR}/sentiment/cnn'
BTC_NEWS_SENTIMENT_DIR = f'{DATA_DIR}/sentiment/bitcoin_news'
GTREND_DIR = f'{DATA_DIR}/crawled_dataset/google_trends'
BINANCE_PRICE_DIR = f'{DATA_DIR}/crawled_dataset/binance_price'
PREDICTION_OUTPUT_DIR = f'{DATA_DIR}/json_price_movement_prediction'
GLASSNODE_API_PATH = ''  # change this with your own
# gsheet related constants
GSERVICE_CREDENTIAL = '/content/sylvan-epoch-255012-ca8e2747c1fa.json'  # change this with your own
GSHEET_NAME = 'ResultTable'  # make sure gservice account above has access to the gsheet
PREDICTION_RESULT_WS_NAME = 'Prediction Result'
TOMORROW_PREDICTION_WS_NAME = 'Tomorrow Prediction'
PREDICTION_HISTORY_WS_NAME = 'Prediction History'
# model constants
MODEL_PARAM_PATH = '/content/drive/MyDrive/CryptoModule/model/optimized_params/0.683_accuracy_2022-03-05.json'
MODEL_THRESHOLD = 0.5
TRAINING_PERIOD = 30  # how many past days are used for training
TODAY_REFERENCE = '2022-02-05'  # set as None to set today's date as reference

# set logging config
logging.basicConfig(format='%(asctime)s - [%(levelname)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

# load data
data_loader = DataLoader(twitter_sentiment_dir=TWITTER_SENTIMENT_DIR,
                         cnn_sentiment_dir=CNN_SENTIMENT_DIR,
                         btcnews_sentiment_dir=BTC_NEWS_SENTIMENT_DIR,
                         gtrend_dir=GTREND_DIR,
                         glassnode_api_path=GLASSNODE_API_PATH,
                         binance_price_dir=BINANCE_PRICE_DIR)
df = data_loader.run(TRAINING_PERIOD, TODAY_REFERENCE)
X_train, X_test, y_train = Utils.split_data(df)
# load model
model = Model()
# feature selection
selected_features = select_features(clf=model.get(), X=X_train[:TRAINING_PERIOD], y=y_train[:TRAINING_PERIOD])
X_with_selected_features = X_train[selected_features]
# HPO
tuned_hyperparams, eval_metrics = fine_tune_model(model.clf, X_with_selected_features, y_train, TRAINING_PERIOD,
                                                  date_column='date', n_trials=20)
print(eval_metrics)

# train and predict
tuned_clf = model.load_param(params=tuned_hyperparams)
tuned_clf.fit(X_train, y_train)
predict_proba = tuned_clf.predict_proba(X_test)
# generate model output
output = generate_output(data_loader, predict_proba, model_threshold=MODEL_THRESHOLD,
                         selected_features=selected_features, eval_metrics=eval_metrics)
print(output)

# insert model output to gsheet
gs_updater = GSheetUpdater(GSERVICE_CREDENTIAL)
gs_updater.update(model_output=output,
                  spreadsheet_name=GSHEET_NAME,
                  prediction_result_ws_name=PREDICTION_RESULT_WS_NAME,
                  tomorrow_prediction_ws_name=TOMORROW_PREDICTION_WS_NAME)

# dump model output as json
Utils.dump_output(output, out_dir=PREDICTION_OUTPUT_DIR)
