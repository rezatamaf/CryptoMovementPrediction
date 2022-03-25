import logging
import datetime as dt
import transformers
from google.colab import drive

from sentiment.model_loader import ModelLoader
from sentiment.sentiment_inferrer import Inferrer
from price_movement.data_loader import DataLoader
from price_movement.util import Utils, GSheetUpdater
from price_movement.price_classifier import Model
from price_movement.output_generator import generate_output
from price_movement.feature_processor import select_features
from price_movement.model_tuner import fine_tune_model
from influence.influence_calculator import InfluenceCalculator

# raw data dirs
DATA_BASE_DIR = '/content/drive/MyDrive/CryptoDataset'
RAW_DATA_DIR = f'{DATA_BASE_DIR}/crawled_dataset'
TWITTER_RAW_DIR = f'{RAW_DATA_DIR}/twitter'
CNN_RAW_DIR = f'{RAW_DATA_DIR}/cnn'
BTC_NEWS_RAW_DIR = f'{RAW_DATA_DIR}/bitcoin_news'
TWITTER_INFLUENCER_RAW_DIR = f'{RAW_DATA_DIR}/twitter_influencers'
TWITTER_INFLUENCER_LIST_DIR = f'{RAW_DATA_DIR}/lunarcrush_influencers'
GTREND_DIR = f'{RAW_DATA_DIR}/google_trends'
BINANCE_PRICE_DIR = f'{RAW_DATA_DIR}/binance_price'
GLASSNODE_API_PATH = f'{DATA_BASE_DIR}/fundamental/glassnode_api_key.json'  # change this with your own

# processed sentiment dirs, [ changed these after testing ]
# TEST_DATA_BASE_DIR = '/content/drive/MyDrive/testCryptoDataset' # remove this after testing
# SENTIMENT_DIR = f'{TEST_DATA_BASE_DIR}/sentiment' # change to DATA_BASE_DIR after testing
SENTIMENT_DIR = f'{DATA_BASE_DIR}/sentiment'
TWITTER_SENTIMENT_DIR = f'{SENTIMENT_DIR}/twitter'
CNN_SENTIMENT_DIR = f'{SENTIMENT_DIR}/cnn'
BTC_NEWS_SENTIMENT_DIR = f'{SENTIMENT_DIR}/bitcoin_news'
TWITTER_INFLUENCER_SENTIMENT_DIR = f'{SENTIMENT_DIR}/twitter_influencers'
INFLUENCE_SCORE_DIR = f'{DATA_BASE_DIR}/influence_score'
PREDICTION_OUTPUT_DIR = f'{DATA_BASE_DIR}/json_price_movement_prediction'

# NLP model constants
RELEVANCE_MODEL_PATH = '/content/drive/MyDrive/CryptoModule/model/model_weight/relevance.hdf5'
TABSA_MODEL_PATH = '/content/drive/MyDrive/CryptoModule/model/model_weight/tabsa.hdf5'
SAMPLE_SIZE = 10000

# gsheet related constants
GSERVICE_CREDENTIAL = '/content/drive/MyDrive/CryptoDataset/gsheet_test/gsheet_cred.json'  # change this with your own
GSHEET_NAME = 'ResultTable'
# GSHEET_NAME = 'Prediction Result'  # make sure gservice account above has access to the gsheet
PREDICTION_RESULT_WS_NAME = 'Prediction Result'
TOMORROW_PREDICTION_WS_NAME = 'Tomorrow Prediction'

# price model constants
MODEL_THRESHOLD = 0.5
TODAY_REFERENCE = dt.datetime.today().date().strftime("%Y-%m-%d")
MODELING_PERIOD = Utils.count_usable_data(TWITTER_SENTIMENT_DIR, TODAY_REFERENCE)  # total days used for modeling
EVALUATION_PERIOD = 60  # part of total days used for HPO evaluation
TEST_PERIOD = 14  # part of total days used for HPO testing

ready = True

# set logging config
logging.basicConfig(format='%(asctime)s - [%(levelname)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
transformers.logging.set_verbosity_error()

# load language model
nlp_model_loader = ModelLoader(relevance_model_path=RELEVANCE_MODEL_PATH,
                               tabsa_model_path=TABSA_MODEL_PATH)
nlp_model_loader.load_weights()

if ready:
    # run sentiment inference
    input_dirs = [BTC_NEWS_RAW_DIR, CNN_RAW_DIR, TWITTER_RAW_DIR, TWITTER_INFLUENCER_RAW_DIR]
    output_dirs = [BTC_NEWS_SENTIMENT_DIR, CNN_SENTIMENT_DIR, TWITTER_SENTIMENT_DIR, TWITTER_INFLUENCER_SENTIMENT_DIR]
    yesterday = dt.datetime.strptime(TODAY_REFERENCE, "%Y-%m-%d") - dt.timedelta(1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    # iterate over dirs and run inference
    for indir, outdir in zip(input_dirs, output_dirs):
        Inferrer().run(nlp_model_loader, indir, outdir, date=yesterday_str, sample_size=SAMPLE_SIZE)
    # run influence calculator
    InfluenceCalculator().run(TWITTER_INFLUENCER_LIST_DIR, TWITTER_INFLUENCER_RAW_DIR,
                              INFLUENCE_SCORE_DIR, date=yesterday_str)

# load data
data_loader = DataLoader(twitter_sentiment_dir=TWITTER_SENTIMENT_DIR,
                         cnn_sentiment_dir=CNN_SENTIMENT_DIR,
                         btcnews_sentiment_dir=BTC_NEWS_SENTIMENT_DIR,
                         gtrend_dir=GTREND_DIR,
                         influence_score_dir=TWITTER_INFLUENCER_RAW_DIR,
                         influencer_sentiment_dir=TWITTER_INFLUENCER_SENTIMENT_DIR,
                         glassnode_api_path=GLASSNODE_API_PATH,
                         binance_price_dir=BINANCE_PRICE_DIR)
df = data_loader.run(MODELING_PERIOD, TODAY_REFERENCE)
X_train, X_test, y_train = Utils.split_data(df)
# load model
model = Model()
# feature selection
selected_features = select_features(clf=model.get(), X=X_train[:-EVALUATION_PERIOD], y=y_train[:-EVALUATION_PERIOD])
X_train_with_selected_features = X_train[selected_features]
X_test_with_selected_features = X_test[selected_features]
# HPO
training_period = MODELING_PERIOD - EVALUATION_PERIOD
tuned_hyperparams, eval_metrics = fine_tune_model(model.clf, X_train_with_selected_features, y_train, training_period,
                                                  holdout_period=TEST_PERIOD, date_column='date', n_trials=20)
print(eval_metrics)

# train and predict
tuned_clf = model.load_param(params=tuned_hyperparams)
tuned_clf.fit(X_train_with_selected_features, y_train)
predict_proba = tuned_clf.predict_proba(X_test_with_selected_features)
# generate model output
output = generate_output(data_loader, predict_proba, model_threshold=MODEL_THRESHOLD,
                         selected_features=selected_features, eval_metrics=eval_metrics)
print(output)

if ready:
    # insert model output to gsheet
    gs_updater = GSheetUpdater(GSERVICE_CREDENTIAL)
    gs_updater.update(model_output=output,
                      spreadsheet_name=GSHEET_NAME,
                      prediction_result_ws_name=PREDICTION_RESULT_WS_NAME,
                      tomorrow_prediction_ws_name=TOMORROW_PREDICTION_WS_NAME)

    # dump model output as json
    Utils.dump_prediction_output(output, out_dir=PREDICTION_OUTPUT_DIR)
