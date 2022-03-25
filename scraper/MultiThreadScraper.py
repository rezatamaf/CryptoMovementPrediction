#!/usr/bin/env python
# coding: utf-8
from scraper.Scraper import TwitterScraper, NewsScraper, GoogleTrendsScraper, BinanceScraper, TokenInfluencersGraber

# Built-in function
from functools import partial
import subprocess
import time

# File manipulation
import pandas as pd
import json
import os

# For Multithreading
import threading

# Progress bar
from tqdm import tqdm

class MultiThreads(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
        
    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def run_all_news_scraper(keyword, news_media=['cnn', 'bitcoin news'],
                         max_pages=1, headless=True, save_to_gdrive=True,
                         directory=['.', '.'], symbol=None, colab=False):
    result = {}
    for media, dirpath in zip(news_media, directory):
        news_scraper = NewsScraper(
            media=media,
            headless=headless,
            random_user_agent=True,
            colab=colab
        )
        
        # Start crawling
        crawled_result = news_scraper.main(
            keyword=keyword,
            max_pages=max_pages,
            save_to_gdrive=True,
            directory=dirpath,
            symbol=symbol
        )
    
def run_all_scraper(news=True, twitter=True, gtrends=True, binance=True, 
                    twitter_crypto_influencer=True, news_media=['cnn', 'bitcoin news'],
                    keyword='bitcoin', multithreads=True, twitter_crypto_influencer_args=None,
                    twitter_args=None, news_args=None, binance_args=None,
                    gtrends_args=None, run_from_colab=False):
    
    # Get credentials json path
    if run_from_colab:
        credentials_json_path = os.path.join('/content/drive/MyDrive/CryptoModule/crawler', 'scraper', 'credentials.json')
    else:
        credentials_json_path = os.path.join('scraper', 'credentials.json')
    
    # Load credentials file
    credentials_file = open(credentials_json_path)      
    credentials = json.load(credentials_file)
    credentials_file.close()
    
    if multithreads:
        all_threads = []
        if news:
            run_all_news_scraper_func = partial(run_all_news_scraper, **news_args, colab=run_from_colab)
            news_scraper_thread = MultiThreads(target=run_all_news_scraper_func, name='news')
            
            # Append thread
            all_threads.append(news_scraper_thread)
            
        if twitter:
            # Initialize twitter scraper    
            twitter_scraper = TwitterScraper(credentials)
            twitter_scraper_func = partial(twitter_scraper.main, **twitter_args, colab=run_from_colab)
            twitter_scraper_thread = MultiThreads(target=twitter_scraper_func)
            
            # Append
            all_threads.append(twitter_scraper_thread)
        
        if binance:
            binance_scraper = BinanceScraper()
            binance_scraper_func = partial(binance_scraper.get_price_df, **binance_args)
            binance_scraper_thread = MultiThreads(target=binance_scraper_func)
            
            # Append
            all_threads.append(binance_scraper_thread)
        
        # Run all threads
        if len(all_threads) > 0:
            for thread in all_threads:
                thread.start()
                
            for thread in all_threads:
                thread.join()
                
        if twitter_crypto_influencer:
            print("[INFO] Gathering influencers tweet ...")
            inf_graber = TokenInfluencersGraber()
            crypto_influencers = inf_graber.get_token_influencers(twitter_crypto_influencer_args['symbol'], limit=300, days=1)
            
            # Generate filename
            crypto_influencers_filename = f"btc_lunarcrush_influencers_{twitter_crypto_influencer_args['end_date']}.csv"
            crypto_influencers_filepath = os.path.join(twitter_crypto_influencer_args['lunarcrush_dir'], crypto_influencers_filename)
            crypto_influencers.to_csv(crypto_influencers_filepath, index=False)
            print("[INFO] Lunarcrush data has been saved.")
            
            # Del argument to avoid error while running TwitterScraper 
            del twitter_crypto_influencer_args['lunarcrush_dir']
            
            # Get influencers list
            crypto_influencers_list = crypto_influencers['twitter_screen_name'].tolist()
            
            if twitter_crypto_influencer_args.get('n_influencers_per_iter'):
                step = twitter_crypto_influencer_args['n_influencers_per_iter']
                del twitter_crypto_influencer_args['n_influencers_per_iter']
                
            else:
                step = 30

            for n_influencer in tqdm(range(0, len(crypto_influencers_list), step)):
                # Generate keyword
                influencer_keyword = [f"from:{influencer}" for influencer in crypto_influencers_list[n_influencer:n_influencer+step]]
                influencer_keyword = ' OR '.join(influencer_keyword)
                twitter_crypto_influencer_args['keyword'] = twitter_crypto_influencer_args['keyword'].split('(')[0]
                twitter_crypto_influencer_args['keyword'] = f"{twitter_crypto_influencer_args['keyword']} ({influencer_keyword})"
                
                ## Start crawl ##
                twitter_scraper = TwitterScraper(credentials)
                twitter_scraper.main(**twitter_crypto_influencer_args, colab=run_from_colab)
                
                # Delay
                time.sleep(3)
                
        if gtrends:
            gtrends_scraper = GoogleTrendsScraper()
            gtrends_scraper.run(**gtrends_args)
            
    else:
        if news:
            run_all_news_scraper(**news_args, colab=run_from_colab)
        
        if twitter:
            twitter_scraper = TwitterScraper(credentials)
            twitter_scraper.main(**twitter_args, colab=run_from_colab)
            
        if twitter_crypto_influencer:
            print("[INFO] Gathering influencers tweet ...")
            inf_graber = TokenInfluencersGraber()
            crypto_influencers = inf_graber.get_token_influencers(twitter_crypto_influencer_args['symbol'], limit=300, days=1)
            
            # Generate filename
            crypto_influencers_filename = f"btc_lunarcrush_influencers_{twitter_crypto_influencer_args['end_date']}.csv"
            crypto_influencers_filepath = os.path.join(twitter_crypto_influencer_args['lunarcrush_dir'], crypto_influencers_filename)
            crypto_influencers.to_csv(crypto_influencers_filepath, index=False)
            print("[INFO] Lunarcrush data has been saved.")
            
            # Del argument to avoid error while running the TwitterScraper 
            del twitter_crypto_influencer_args['lunarcrush_dir']
            
            # Get influencers list
            crypto_influencers_list = crypto_influencers['twitter_screen_name'].tolist()
            
            if twitter_crypto_influencer_args.get('n_influencers_per_iter'):
                step = twitter_crypto_influencer_args['n_influencers_per_iter']
                del twitter_crypto_influencer_args['n_influencers_per_iter']
                
            else:
                step = 30

            for n_influencer in tqdm(range(0, len(crypto_influencers_list), step)):
                # Generate keyword
                influencer_keyword = [f"from:{influencer}" for influencer in crypto_influencers_list[n_influencer:n_influencer+step]]
                influencer_keyword = ' OR '.join(influencer_keyword)
                twitter_crypto_influencer_args['keyword'] = twitter_crypto_influencer_args['keyword'].split('(')[0]
                twitter_crypto_influencer_args['keyword'] = f"{twitter_crypto_influencer_args['keyword']} ({influencer_keyword})"
                
                ## Start crawl ##
                twitter_scraper = TwitterScraper(credentials)
                twitter_scraper.main(**twitter_crypto_influencer_args, colab=run_from_colab)
                
                # Delay
                time.sleep(3)
            
        if gtrends:
            gtrends_scraper = GoogleTrendsScraper()
            gtrends_scraper.run(**gtrends_args)
            
        if binance:
            binance_scraper = BinanceScraper()
            binance_scraper.get_price_df(**binance_args)

