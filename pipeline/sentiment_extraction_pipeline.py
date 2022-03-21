import logging

from sentiment.model_loader import ModelLoader
from sentiment.sentiment_inferrer import Inferrer

# crawled dir
RAW_DATA_DIR = '/content/drive/MyDrive/CryptoDataset/crawled_dataset'
TWITTER_RAW_DIR = f'{RAW_DATA_DIR}/twitter'
CNN_RAW_DIR = f'{RAW_DATA_DIR}/cnn'
BTC_NEWS_RAW_DIR = f'{RAW_DATA_DIR}/bitcoin_news'
INFLUENCER_RAW_DIR = f'{RAW_DATA_DIR}/twitter_influencers'
# sentiment output dir, changed these after testing
SENTIMENT_DIR = '/content/drive/MyDrive/test_CryptoDataset/sentiment'
TWITTER_SENTIMENT_DIR = f'{SENTIMENT_DIR}/twitter'
CNN_SENTIMENT_DIR = f'{SENTIMENT_DIR}/cnn'
BTC_NEWS_SENTIMENT_DIR = f'{SENTIMENT_DIR}/bitcoin_news'
INFLUENCER_SENTIMENT_DIR = f'{SENTIMENT_DIR}/twitter_influencers'
# model path
RELEVANCE_MODEL_PATH = '/content/drive/MyDrive/CryptoModule/model/model_weight/relevance.hdf5'
TABSA_MODEL_PATH = '/content/drive/MyDrive/CryptoModule/model/model_weight/tabsa.hdf5'
TODAY_REFERENCE = '2022-02-05'

logging.basicConfig(format='%(asctime)s - [%(levelname)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

# load model
nlp_model_loader = ModelLoader(relevance_model_path=RELEVANCE_MODEL_PATH,
                               tabsa_model_path=TABSA_MODEL_PATH)
nlp_model_loader.load_weights()

# run inference
input_dirs = [BTC_NEWS_RAW_DIR, CNN_RAW_DIR, TWITTER_RAW_DIR, INFLUENCER_RAW_DIR]
output_dirs = [BTC_NEWS_SENTIMENT_DIR, CNN_SENTIMENT_DIR, TWITTER_SENTIMENT_DIR, INFLUENCER_SENTIMENT_DIR]
# iterate over dirs and run inference
for indir, outdir in zip(input_dirs, output_dirs):
    # leave date argument to run the inference on today's date
    Inferrer().run(nlp_model_loader, indir, outdir, date=TODAY_REFERENCE, sample_size=5)
