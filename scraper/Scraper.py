#!/usr/bin/env python
# coding: utf-8

# File system
import subprocess
import json
import sys
import os
import re

try:
    import selenium
except Exception as e:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'selenium'])
    import selenium

from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver import Firefox, FirefoxOptions, FirefoxProfile
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException

# UDF
from scraper.TwitterAPIPreprocessing import TwitterPreprocessing

# Common packages
from functools import partial
import pandas as pd
import numpy as np
import requests
import random
import re

# Datetime packages
from dateutil.relativedelta import relativedelta
import datetime as dt
import time

# Install lunarcrush
subprocess.check_output(['pip', 'install', 'lunarcrush'])
from lunarcrush import LunarCrush

# Random User
try:
    import random_user_agent
except Exception as e:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'random_user_agent'])
    import random_user_agent
    
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

# Google trends API
try:
    from pytrends import dailydata
except Exception as e:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytrends'])
    from pytrends import dailydata
    
# Multithreads
import threading

# GDrive 
from scraper.Google import GDrive

class TwitterScraper():
    """
    TwitterScrapper is scrapper to scrape twitter data
    """
    def __init__(self, credentials):
        # Set Bearer Token
        os.environ['TOKEN'] = credentials['BearerToken']

    def __auth(self):
        return os.getenv('TOKEN')

    def __create_headers(self, bearer_token):
        '''
        Function to create headers using bearer token
        '''
        headers = {
            # "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36",
            "Authorization": "Bearer {}".format(bearer_token)
        }
        return headers

    def __create_url(self, keyword, start_date, end_date, by='all', tweets_per_request=10):
        """
        Build endpoint

        parameters:
            keyword = Twitter keyword we want to search
            start_date = start datetime to scrap (timestamps) using ISO 8601/RFC 3339 format YYYY-MM-DDTHH:mm:ssZ
            end_date = the end of the date to end scraping (timestamps) using ISO 8601/RFC 3339 format YYYY-MM-DDTHH:mm:ssZ
            tweets_per_request = max results limited between 10 - 500 results. To add more than 500, "next_token" comes into the place
        """

        # Endpoint
        search_url = f"https://api.twitter.com/2/tweets/search/{by}" 

        # Endpoint parameters
        query_params = {
            'query': keyword,
            'start_time': start_date,
            'end_time': end_date,
            'max_results': tweets_per_request,
            'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
            'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
            'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
            'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
            'next_token': {}
        }

        return (search_url, query_params)

    def __connect_to_endpoint(self, url, headers, params, next_token = None):
        '''
        Parameters:
            url = endpoint
            headers = headers using bearer token
            params = query_params from create_url function returns

        Returns:
            response.json() = data that have(s) been gathered from "GET" request
        '''
        #params object received from create_url function
        params['next_token'] = next_token   

        # Request "GET"
        response = requests.request("GET", url, headers = headers, params = params)

        # Show request response
        print("Endpoint Response Code: " + str(response.status_code))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()
    
    def search(self, keyword, blacklist_words=[], lang='id',
               start_date="2021-03-01", end_date="2021-03-31",
               start_time=None, end_time=None,
               tweets_per_request=10, max_count=10, search_by='all'):
        """
        parameters:
            keyword = Twitter keyword we want to search
            lang = tweet language to scrape
            start_date = start datetime to scrap (timestamps) using ISO 8601/RFC 3339 format YYYY-MM-DDTHH:mm:ssZ
            end_date = the end of the date to end scraping (timestamps) using ISO 8601/RFC 3339 format YYYY-MM-DDTHH:mm:ssZ
            tweets_per_request = max results limited between 10 - 500 results. To add more than 500, "next_token" comes into the place
            max_count = maximum number of data to retrieve
        """
        # Authentication
        if tweets_per_request < 10:
            raise ValueError('Max results must be more than or equal to 10')
            
        bearer_token = self.__auth()
        
        # Create Headers
        headers = self.__create_headers(bearer_token)
        
        # Search parameters
        blacklist_words = ''.join([''.join(["-"+word+" " for word in text]) for text in blacklist_words])
        keyword_search = f"{keyword} {blacklist_words}lang:{lang}"
        
        if start_time:
            start_time = re.sub('.000', '', start_time)
            start_time = re.sub('[./TZ]', ":", start_time)
            start_time = "T" + start_time + ".000Z"
            start_time = start_date + start_time
            
        else:
            start_time = start_date+"T00:00:00.000Z"
            
        if end_time:
            end_time = re.sub('.000', '', end_time)
            end_time = re.sub('[./TZ]', ":", end_time)
            end_time = "T" + end_time + ".000Z"
            end_time = end_date + end_time
            
        else:
            end_time = end_date+"T23:59:59.000Z"
        tweets_per_request = tweets_per_request
        next_token = None

        
        # Rule base
#         data = []
        flag = True
        count = 0
        max_count = max_count
        total_tweets = 0
        
        print()
        print('Keyword to search:', keyword)
        print('Number of data to search:', max_count)
        print('Start date time :', start_time)
        print('End date time   :', end_time)
        print('Starting to retrieve tweet ...')
        print('================================')
        # Show first token
        print('Next token:', next_token)
        print("Total # of tweets retrieved:", total_tweets)
        print("================================")
        while flag:
            # Rule base check (while loop will be stopped when count >= max_count)
            if count >= max_count:
                break
            
            # Build and connect endpoint
            url = self.__create_url(keyword_search, start_time, end_time,
                                    tweets_per_request=tweets_per_request, by=search_by)
            json_response = self.__connect_to_endpoint(url[0], headers, url[1], next_token)
            
            yield json_response
        
            # Show number of data retrieved
            result_count = json_response['meta']['result_count']
            
            if 'next_token' in json_response['meta']:
                # Save the token for the next call
                next_token = json_response['meta']['next_token']
                print('Next token:', next_token)
                
                # Check if result_count increasing and next_token is not none
                if result_count is not None and result_count > 0 and next_token is not None:
                    count += result_count
                    total_tweets += result_count
                    print("Total # of tweets retrieved:", total_tweets)
                    print("================================")
                    
                    # Sleep (to avoid RTO)
                    time.sleep(5)
                    
            else:
                if result_count is not None and result_count > 0:
                    count += result_count
                    total_tweets += result_count
                    print("Total # of tweets retrieved:", total_tweets)
                    print("================================")

                # Stop loop
                print("There's no data to retrieve.")
                print("Execution stopped.\n")
                flag = False
                next_token = None
        
        print()
        print("================================")
        print("============  DONE  ============")
        print("================================")
    
    def _save_as_dataframe(self, data, directory='.', symbol=None, save_by_date=True, do_all_option=None):
        # Get all dates
        if save_by_date:
            data['date'] = pd.to_datetime(data['date'])
            all_dates = data['date'].dt.date.unique()
            
            # Options
            overwrite_count = 0
            append_count = 0
            # do_all_option = None
            for date in all_dates:
                subset = data[(data['date'].dt.date>=date) & (data['date'].dt.date<=date)]
                
                # Drop duplicates if any
                subset = subset.drop_duplicates().reset_index(drop=True)
    
                # Save to local
                if symbol:
                    filename = f"{symbol}_twitter_{date}"
                    
                else:
                    filename = f"{keyword.split('-')[0].strip()}_twitter_{date}"
                    
                filename = filename+'.csv'
                local_filename = os.path.join(directory, filename)
                
                if os.path.exists(local_filename):
                    while True:
                        if not do_all_option:
                            selected_option = str(input("\n[WARNING] Found similar file name while saving Twitter data.\n1) Overwrite\n2) Append\n3) Save with unique name\n0) Cancel\nChoose one (1/ 2/ 3/ 0): "))
                            
                            if selected_option == "1":
                                subset.to_csv(local_filename, mode='w', index=False)
                                print('[INFO] File (.csv) has been saved by overwriting it.')
                                overwrite_count += 1
                                if overwrite_count >= 2:
                                    print('\n[WARNING] Do you want to do all duplicated files to overwrite?')
                                    selected_option = str(input("0) Yes,\n1) No\nChoose one: "))
                                    if selected_option == '0':
                                        do_all_option = 'overwrite'
                                break
                                
                            elif selected_option == "2":
                                # Read existing data
                                similar_data = pd.read_csv(local_filename)
                                subset = pd.concat([similar_data, subset], axis=0, ignore_index=True)
                                
                                # Save
                                subset.to_csv(local_filename, mode='w', index=False)
                                print('[INFO] Data has been appended to an existing file (.csv).')
                                append_count += 1
                                if append_count >= 2:
                                    print('\n[WARNING] Do you want to do all duplicated files to be appended?')
                                    selected_option = str(input("0) Yes,\n1) No\nChoose one: "))
                                    if selected_option == '0':
                                        do_all_option = 'append'
                                    break
                                break
                        
                            elif selected_option == "3":
                                # Find unique name
                                count = 1
                                while True:
                                    new_local_filename = f"({count}) " + filename
                                    new_local_filename = os.path.join(directory, new_local_filename)
                                    if os.path.exists(new_local_filename):
                                        count += 1
                                    
                                    else:
                                        subset.to_csv(new_local_filename)
                                        print('[INFO] File (.csv) has been saved with unique name.')
                                        break
                                
                                break
                            
                            elif selected_option == "0":
                                confirmation_option = str(input("[WARNING] Data will not be saved, are you sure? (y/n):"))
                                
                                if confirmation_option.lower() == 'y' or confirmation_option == '':
                                    print("[INFO] Saving data has been canceled.")
                                    break
                                
                                else:
                                    print()
                                
                            else:
                                print('[WARNING] Please only write the number from the provided options. example for overwrite: 1\n')
                        
                        else:
                            if do_all_option == 'overwrite':
                                subset.to_csv(local_filename, mode='w', index=False)
                                print(f'[INFO] File {filename} has been saved by overwriting it.')
                                break
                            
                            elif do_all_option == 'append':
                                # Read existing data
                                similar_data = pd.read_csv(local_filename)
                                subset = pd.concat([similar_data, subset], axis=0, ignore_index=True)
                                
                                # Save
                                subset.to_csv(local_filename, mode='w', index=False)
                                print(f'[INFO] {filename} has been appended to an existing file (.csv).')
                                break
            
                else:
                    subset.to_csv(local_filename, mode='w', index=False)
                    print('[INFO] File (.csv) has been saved.')
            
        # Else if: save_by_date FALSE
        else:
            if symbol:
                filename = f"{symbol}_twitter"
                filename = filename+'.csv'
            
            else:
                filename = f"{keyword.split('-')[0].strip()}_twitter_{date}"
                filename = filename+'.csv'
            
            local_filename = os.path.join(directory, filename)
            data.to_csv(local_filename, index=False)
    
    def _save_as_json(self, data, directory='.', symbol=None):
        # Save to local
        if symbol:
            filename = f"{symbol}_twitter"
            
        else:
            filename = f"{keyword.split('-')[0].strip()}_twitter"
            
        filename = filename+'.json'
        local_filename = os.path.join(directory, filename)
        
        # Save as json
        with open(local_filename, 'w') as f:
            json.dump(data, f, indent=4)
            
        print('File (.json) has been saved.')
    
    def main(self, keyword, max_data, blacklist_words=[], start_date='2022-01-16', 
             end_date='2022-01-18', start_time=None, end_time=None,
             lang='en', save_as_dataframe=True, directory='.', do_all_option=None,
             save_to_gdrive=True, colab=False, symbol=None, save_by_date=True):
        # Init Twitter Preprocessing
        preprocessor = TwitterPreprocessing()

        data = []
        try:
            for tweet in self.search(keyword, 
                                         tweets_per_request=500, 
                                         max_count=max_data,
                                         blacklist_words=blacklist_words,
                                         lang=lang,
                                         start_date=start_date,
                                         end_date=end_date,
                                         start_time=start_time,
                                         end_time=end_time):
                
                if len(tweet) > 0:
                    if save_as_dataframe:
                        # If merge False, output will be 3: tweet, user, place
                        # If merge True, output only one: combined data consists of: tweet, user, place
                        tweet = preprocessor.json_to_dataframe(tweet, merge=True)
        
                        # Fix public metrics column format
                        tweet = preprocessor.extract_public_metrics(tweet)
                        
                    # Append every 'n' request to list (named=data)
                    data.append(tweet)
                
        except Exception as e:
            # Get last data
            if len(data) > 0:
                temp_data = data[-1].copy()
                temp_data['date'] = pd.to_datetime(temp_data['date'])
                    
                print(e)
                print(f"Stopped at {temp_data['date'].min()}")
                
            else:
                print(e)
            
        finally:
            if len(data) > 0:
                if save_as_dataframe:
                    data = pd.concat(data, ignore_index=True)
                    data['article'] = data['article'].str.replace('\r', ' ')
                    self._save_as_dataframe(data, directory=directory, symbol=symbol, save_by_date=save_by_date, do_all_option=do_all_option)
                    
                    
                else:
                    self._save_as_json(data, directory=directory, symbol=symbol)
                    
                if save_to_gdrive:
                    if not colab:
                        gdrive = GDrive()
            
                        # Upload saved file to gdrive
                        twitter_folder_id = '1DULfTw8eKjTyhsPHvWMana-cxicrlFLh'
                        gdrive.upload(filename=filename, directory_id=twitter_folder_id, local_dir=directory)
            
                        # Delete file
                        os.remove(local_filename)

def colab_webdriver_install():
    if not os.path.exists('/usr/lib/chromium-browser'):
        # to update ubuntu to correctly run apt install
        subprocess.call('apt-get update', shell=True)
        subprocess.call('apt install chromium-chromedriver', shell=True)
        subprocess.call('cp /usr/lib/chromium-browser/chromedriver /usr/bin', shell=True)
        sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
    
class NewsScraper:
    stale = 0
    media = ''
    news_links = []
    search_url = ''
    ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
    
    def __init__(self, geckodriver_path='./geckodriver', colab=False,
                 headless=False, disable_media=True, block_popup=True,
                 media='bbc', random_proxy=False, random_user_agent=False):
        
        # Set geckodriver path
        self.service = Service(geckodriver_path)

        # Create options class
        if colab:
            colab_webdriver_install()
            self.options = ChromeOptions()
            
            # Mandatory setting
            self.options.add_argument('--headless')
            self.options.add_argument('--no-sandbox')
            self.options.add_argument('--disable-dev-shm-usage')
            
        else:
            self.options = FirefoxOptions()
        
        # Headless webdriver
        if headless:
            self.options.add_argument('--headless')
        
        # Disable load photo
        if disable_media:
            if colab:
                self.options.add_experimental_option('prefs', {"profile.managed_default_content_settings.images": 2})
                
            else:
                self.options.set_preference('permissions.default.image', 2)
                self.options.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
            
        if block_popup:
            if colab:
                self.options.add_argument("--disable-notifications")
                self.options.add_argument("--disable-popup-blocking")
                
            else:
                self.options.set_preference("dom.popup_maximum", 0)
            
        if random_proxy:
            # Indo Proxies
            proxies = ['203.142.68.141:3128', '202.51.106.229:8080', '103.253.113.54:80']
            PROXY = random.choice(proxies)
            capabilities = DesiredCapabilites.FIREFOX
            
            prox = Proxy()
            prox.proxy_type = ProxyType.MANUAL
            prox.auto_detect = False
            prox.http_proxy = PROXY
            prox.ssl_proxy = PROXY
            prox.add_capabilities(capabilities)
            
        if random_user_agent:
            software_names = [SoftwareName.FIREFOX.value]
            operating_systems = [OperatingSystem.WINDOWS.value,
                               OperatingSystem.LINUX.value]
            user_agent_rotator = UserAgent(software_names=software_names,
                                           operating_names=operating_systems,
                                           limit=100)
            self.user_agent = user_agent_rotator.get_random_user_agent()

        # Initialize Firefox webdriver
        if colab:
            self.driver = Chrome('chromedriver', options=self.options)
        else:
            self.driver = Firefox(service=self.service, options=self.options)
        
        # Global setting
        self.media = media
        self.colab = colab
        
        if self.media == 'bbc':
            # Redirect to BBC news home page
            self.driver.get('https://www.bbc.com/')
            
        elif self.media == 'cnn':
            self.driver.get('https://edition.cnn.com/')
            
    def _convert_relative_datetime(self, retrieved_at, value):
        satuan = ['second', 'min', 'minute', 'hour', 'day',
                'week', 'month', 'year']
        
        for satuan_waktu in satuan:
            relative_date_pattern = f'(\d+) {satuan_waktu}s? ago'
            digit = re.findall(relative_date_pattern, value)
            if digit:
                digit = digit[0]
                if satuan_waktu == 'second':
                    fixed_datetime = dt.timedelta(seconds=int(digit))
                    
                elif satuan_waktu in ['min', 'minute']:
                    fixed_datetime = dt.timedelta(minutes=int(digit))
                    
                elif satuan_waktu == 'hour':
                    fixed_datetime = dt.timedelta(hours=int(digit))
                
                elif satuan_waktu == 'day':
                    fixed_datetime = dt.timedelta(days=int(digit))
                
                elif satuan_waktu == 'week':
                    fixed_datetime = dt.timedelta(days=7*int(digit))
                    
                elif satuan_waktu == 'month':
                    fixed_datetime = relativedelta(months=int(digit))
                    
                elif satuan_waktu == 'year':
                    fixed_datetime = relativedelta(years=int(digit))
                
                return retrieved_at - fixed_datetime
        
        return value
    
    def __search(self, keyword, page=1):
        if self.media == 'bbc':
            # Set keyword
            self.search_url = f'https://www.bbc.co.uk/search?q={keyword}&page={page}'
            
        if self.media == 'cnn':
            # Set search url
            result_per_page = 10
            result_size = (page - 1) * result_per_page
            self.search_url = f'https://edition.cnn.com/search?q={keyword}&size={result_per_page}&from={result_size}&page={page}'
        
        if self.media == 'bitcoin news':
            # Set keyword
            keyword = '+'.join(keyword.split())
            self.search_url = f'https://news.bitcoin.com/page/{page}/?s={keyword}'
            
        # Go to the selected search url
        self.driver.get(self.search_url)

    def __get_bbc_content(self):
        try:
            while True:
                try:
                    paragraphs_xpath = WebDriverWait(
                        self.driver, 5,
                        ignored_exceptions=self.ignored_exceptions).until(
                                EC.visibility_of_any_elements_located(
                                    (By.XPATH,
                                     '//div[@data-component="text-block"] | //p[contains(@data-reactid, "paragraph")]')
                                )
                            )
                    news_headline = WebDriverWait(self.driver, 20).until(
                        EC.visibility_of_element_located(
                            (By.TAG_NAME, 'h1')
                        )
                    ).text
                    
                    if paragraphs_xpath:
                        # Extract all paragraph text
                        paragraphs = []
                        for paragraph_xpath in paragraphs_xpath:
                            paragraphs.append(paragraph_xpath.text)

                        paragraphs = ' '.join(paragraphs)

                        return (news_headline, paragraphs, self.media)

                    else:
                        return (np.nan, np.nan, np.nan)
                    
                except StaleElementReferenceException:
                    pass
            
        except TimeoutException:
            return (np.nan, np.nan)
        
    def __bbc_content(self, keyword, max_pages=1):        
        for page in range(1, max_pages+1):
            print(f'[INFO] Into {keyword} page {page}.')
            self.__search(keyword, page=page)
            print('[INFO] Done.')
            
            # Search result links
            links = self.driver.find_elements(by='xpath', value='//ul/li/div/div/div/div/a')

            # Get all links and store into variable named news_links
            self.news_links = []
            for link in links:
                self.news_links.append(link.get_attribute('href'))

            for link in self.news_links:
                self.driver.get(link)
                result = self.__get_bbc_content()
                yield result
            
            print(f'[INFO] {page} done.')
            
    def __cnn_content(self, keyword, max_pages=1):
        total_news_collected = 0
        for page in range(1, max_pages+1):
            self.__search(keyword, page=page)
            
            # Get results contents
            search_results = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_any_elements_located(
                    (By.CLASS_NAME, 'cnn-search__result-contents')
                )
            )
            
            for content in search_results:
                content_body = content.text
                try:
                    title, news_datetime, article = content_body.split('\n')
                    
                    # Send log about number of data that has been collected
                    total_news_collected += 1
                    print(f'[INFO] News collected: {total_news_collected}', end='\r')
                    
                    yield (news_datetime, title, article, self.media)
                    
                except ValueError:
                    yield (np.nan, np.nan, np.nan, self.media)
                    
    def __get_bitcoin_news_content(self):
        while True:
            try:
                title = self.driver.find_element(by='xpath', value='//h1[@itemprop="headline"]').text
                relative_time = self.driver.find_element(by='class name', value='article__info__date').text
                date_scrape = dt.datetime.now()
                paragraphs_xpath = WebDriverWait(
                                    self.driver, 20,
                                    ignored_exceptions=self.ignored_exceptions
                                ).until(
                                    EC.visibility_of_any_elements_located(
                                        (By.TAG_NAME, "p")
                                    )
                            )

                paragraphs = []
                for paragraph in paragraphs_xpath:
                    paragraphs.append(paragraph.text)

                article = ('\n'.join(paragraphs)).split('Image Credits')[0]
                return (relative_time, date_scrape, title, article, self.media)
            
            except StaleElementReferenceException:
                self.stale += 1
                pass
            
    def __bitcoin_news_content(self, keyword, max_pages=5):
        print(f'[INFO] Total data that will be collected: {max_pages*14}')
        
        total_news_collected = 0
        for num_page in range(1, max_pages+1):
            self.__search(keyword, page=num_page)

            # Get all results data
            news_result = self.driver.find_elements(by='xpath', value='//div[contains(@class, "td_module_16")]')
            news_link_xpath = self.driver.find_elements(by='xpath', value="//div[@class='td-module-thumb']/a")

            news_links = []
            for news_link in news_link_xpath:
                news_links.append(news_link.get_attribute('href'))
            
            for news_link in news_links:
                # Get to each result
                self.driver.get(news_link)
                
                # Collect article data
                article = self.__get_bitcoin_news_content()
                
                # Send log about number of data that has been collected
                total_news_collected += 1
                print(f'[INFO] Stale emerge: {self.stale} | News collected: {total_news_collected}', end='\r')
                yield article
        
    def run(self, keyword='bitcoin', max_pages=1):
        if self.media == 'bbc':
            for data in self.__bbc_content(keyword=keyword, max_pages=max_pages):
                yield data
                
        elif self.media == 'cnn':
            for data in self.__cnn_content(keyword=keyword, max_pages=max_pages):
                # Store and convert data list into dataframe
                data = pd.DataFrame([data], columns=['date', 'title', 'article', 'media'])
                yield data
                
        elif self.media == 'bitcoin news':
            for data in self.__bitcoin_news_content(keyword=keyword, max_pages=max_pages):
                # Store and convert data list into dataframe
                data = pd.DataFrame([data], columns=['relative_date', 'scrape_date', 'title', 'article', 'media'])
                
                # Fix relative date by converting it into absolute datetime
                data['date'] = data.apply(lambda row: self._convert_relative_datetime(
                    row['scrape_date'], row['relative_date']), axis=1)
                
                # Trim first paragraph and two last paragraphs
                data['article'] = data['article'].apply(lambda x: '\n'.join(x.split('\n')[1:-2]))
                
                yield data[['date', 'title', 'article', 'media']]
        
        else:
            print('Sorry, selected media is not provided in this scraper.')
            print('These are provided medias: 1) BBC, 2) CNN 3) Bitcoin News. Please select one of them.')
            
        self.driver.close()
        
    def main(self, keyword='bitcoin', symbol=None, max_pages=1, save_to_gdrive=True, directory='.'):
        all_data = []
        for data in self.run(keyword=keyword, max_pages=max_pages):
            all_data.append(data)

        all_data = pd.concat(all_data, ignore_index=True)
        all_data['date'] = pd.to_datetime(all_data['date'])
        
        if self.media == 'bitcoin news':
            all_data['date'] = all_data['date'].dt.date
            
        # Drop if there's any duplicate
        all_data = all_data.drop_duplicates()
        
        # Get all unique dates
        if self.media == 'bitcoin news':
            all_dates = all_data['date'].unique()
            
        else:
            all_dates = all_data['date'].dt.date.unique()
        for date in all_dates:
            if self.media == 'bitcoin news':
                subset = all_data[(all_data['date']>=date)&(all_data['date']<=date)]
                
            else:
                subset = all_data[(all_data['date'].dt.date>=date)&(all_data['date'].dt.date<=date)]
            
            # Get file name
            merged_media_name = "_".join(self.media.split())
            
            if symbol:
                filename = f'{symbol}_{merged_media_name}_{date}.csv'
            else:
                filename = f'{keyword}_{merged_media_name}_{date}.csv'
            
            # Save the file
            local_filename = os.path.join(directory, filename)
            subset.to_csv(local_filename, index=False)
            print(f'Data from {self.media} at {date} has been saved successfully.')
        
        if save_to_gdrive:
            if not self.colab:
                gdrive = GDrive()
                
                # Get news_folder_id
                if self.media == 'cnn':
                    news_folder_id = '17dPD324fi9-LJj9wt_0FlA9k4bXcHD7m'
                    
                elif self.media == 'bitcoin news':
                    news_folder_id = '1j5TUcWILXzL1wPm3fKvj1kdBFIPU-lvB'
                    
                else:
                    news_folder_id = '17dPD324fi9-LJj9wt_0FlA9k4bXcHD7m'
                
                # Upload saved file to gdrive
                gdrive.upload(filename=filename, directory_id=news_folder_id, local_dir=directory)
    
                # Delete file
                os.remove(local_filename)

class GoogleTrendsScraper:
  def run(self, keyword, start_date, end_date, symbol=None, save_file=True, directory=None):
    if type(start_date) == str:
      start_year, start_month = re.split('[-/:]', start_date)[:2]
      start_year = int(start_year.strip())
      if start_month.startswith('0'):
        start_month = int(start_month.strip()[1])

      else:
        start_month = int(start_month.strip())

    if type(end_date) == str:
      end_year, end_month = re.split('[-/:]', end_date)[:2]
      end_year = int(end_year.strip())
      if end_month.startswith('0'):
        end_month = int(end_month.strip()[1])

      else:
        end_month = int(end_month.strip())

    print('[INFO] Getting google trends data ..', end='\r')
    result = dailydata.get_daily_data(keyword, start_year, start_month, end_year, end_month)
    result = result.reset_index()
    print('[INFO] Google trends data gathered.')

    # Convert to string
    result['date'] = result['date'].astype(str)

    # Rename columns
    result.rename(columns={keyword.strip(): "trends"}, inplace=True)

    if save_file:
      print('[INFO] Saving the data.')
      
      # Check if last row has 0 value
      last_row = result["trends"].iloc[-1]
      if last_row == 0:
        result = result.iloc[:-1]
         
      # Convert dataframe into json format
      count = 0
      result_json = result[['date', 'trends']].to_dict('records')
      for json_data in result_json:
        # Generate filename
        if symbol:
          filename = f"{symbol.strip()}_trend_{json_data['date']}.json"
        else:
          filename = f"{keyword.strip()}_trend_{json_data['date']}.json"
        
        # Join with directory path
        local_filename = os.path.join(directory, filename)
        
        # Check if local_filename already exist
        if not os.path.exists(local_filename):
            with open(local_filename, 'w') as json_f:
              json.dump(json_data, json_f, indent=4)
              print(f'[INFO] new file saved: {local_filename}.')
              count += 1
        
      if count < 1:
          print('[INFO] No data saved.')

    else:
      return result

class BinanceScraper:
    def get_price_df(self, symbol='BTCUSDT', interval='1d', n_days=3, directory=None, symbol_name='btc'):
      url = 'https://api.binance.com/api/v3/klines?symbol=' + \
          symbol + '&interval=' + interval + '&limit='  + str(n_days+1)
      df = pd.read_json(url)
      df.columns = ['opentime', 'open', 'high', 'low', 'close_price', 'volume', 'closetime', 'quote_asset_volume', 'number_of_trades','taker_by_base', 'taker_buy_quote', 'ignore']  
      df['date'] = df.closetime.apply(lambda x: dt.datetime.utcfromtimestamp(int(x)/1000).date())
      df['date'] = df['date'].astype(str)
      binance_json = df[['date', 'close_price']].to_dict('records')
      for json_f in binance_json:
          filename = f"close_price_{json_f['date']}.json"
          local_filename = os.path.join(directory, filename)
          with open(local_filename, 'w') as f:
            json.dump(json_f, f, indent=4)
      
      print('[INFO] All binance data gathered.')
      
class TokenInfluencersGraber:
  def __init__(self):
    self.lc = LunarCrush()
  
  def get_token_influencers(self, symbol: str, days=1, limit=200) -> pd.DataFrame:
    btc_influencers = self.lc.get_influencers(
        symbol=[symbol], days=days, order_by='influential', limit=limit
        )
    
    # Convert to dataframe
    btc_influencers = pd.DataFrame(btc_influencers['data'])
    btc_influencers['token'] = symbol

    # Sort data
    btc_influencers = btc_influencers.sort_values(
        'influencer_rank_average', 
        ascending=False
        )
    
    btc_influencers = btc_influencers.reset_index(drop=True)

    return btc_influencers
