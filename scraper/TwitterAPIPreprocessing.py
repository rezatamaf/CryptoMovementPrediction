# Common packages
import pandas as pd
import numpy as np
import json
import sys
import ast
import re

# Setup
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class TwitterPreprocessing:
    def json_to_dataframe(self, json_file, merge=False):
        """
        Fungsi untuk mengkonversikan json menjadi DataFrame
        
        Returns:
            base = Data utama yang berisikan informasi tweet
            users = Data user yang terlibat dalam percakapan tweet
            places = Data tempat user yang terlibat dalam percakapan tweet
        """
        normalized_df = pd.json_normalize(json_file)
        base = pd.concat([df for df in normalized_df['data'].apply(lambda x: pd.DataFrame(x))],
                             axis=0)
        users = pd.concat([df for df in normalized_df['includes.users'].apply(lambda x: pd.DataFrame(x))],
                             axis=0)
        
        try:
            places = pd.concat([df for df in normalized_df['includes.places'].dropna(how='all').apply(lambda x: pd.DataFrame(x))],
                                 axis=0)
                                 
        except KeyError:
            places = pd.DataFrame()

        # Rename Index on base dataframe into tweet_id
        base.rename(columns={'id':'tweet_id'}, inplace=True)

        # Drop duplicates
        users = users.drop_duplicates(subset=['id'])
        places = places.drop_duplicates(subset=['id'])

        # Extract tweet type (quote/ replied to/ retweet)
        if 'referenced_tweets' in base.columns:
            base['tweet_type'] = base['referenced_tweets'].fillna(0).apply(lambda x: x[0]['type'] 
                                                                           if x != 0 else np.nan)
            cols_to_del = ['referenced_tweets']
                                                                           
        else:
            base['tweet_type'] = np.nan
            cols_to_del = []

        try:
            # Extract geo id
            base['geo_id'] = base['geo'].fillna(0).apply(lambda x: x['place_id'] 
                                                         if x != 0 else np.nan)
            cols_to_del.append('geo')
            
        except KeyError:
            pass

        # Drop unnecessary columns
        unnecessary_cols = [
            'lang',
            'reply_settings',
            'source',
            'conversation_id'
        ] + cols_to_del
        
        # Drop all unnecessary columns
        base.drop(columns=unnecessary_cols, axis=1, inplace=True)

        if merge:
            ## MERGE INTO ONE ##
            # Get author user information
            twitter = pd.merge(base, 
                               users[['id', 'username']], 
                               left_on='author_id', 
                               right_on='id', 
                               how='left')

            # Initialize empty list to store column(s) to del that generated by merge treatment
            cols_to_del = []
            
            # Get target user information
            try:
                twitter = pd.merge(twitter, 
                                   users[['id', 'username']], 
                                   left_on='in_reply_to_user_id', 
                                   right_on='id', 
                                   how='left',
                                   suffixes=('_source', '_target'))
                                   
            except KeyError:
                pass
                
	    # Get user's place information
            try:
                twitter = pd.merge(twitter,
                                   places[['id', 'country_code', 'country', 'full_name']],
                                   left_on='geo_id',
                                   right_on='id',
                                   how='left')
                
                # List out unnecessary column generated by merge treatment
                cols_to_del.append('id')
                
            except:
                pass
            
            # Rename all necessary columns
            twitter.rename(columns={'created_at': 'date', 'text': 'article'}, inplace=True)
            
            # Add empty columns
            twitter['media'] = 'twitter'
            twitter['headline'] = np.nan
            
            # Check if withheld is part of twitter col
            if 'withheld' in twitter.columns:
                cols_to_del.append('withheld')
            
            # Drop unnecessary column(s) generated by merge treatment
            if cols_to_del:
                twitter = twitter.drop(cols_to_del, axis=1)
            
            return twitter
        
        else:
            return (base, users, places)
        
    def extract_mentions(self, data_source):
        '''Extract all username that starts with at (@)
        '''
        pattern = r'(@\w+)'
        return data_source.apply(lambda x: re.findall(pattern, x) if len(re.findall(pattern, x)) > 0 else np.nan)
                                 
    def extract_hashtags(self, data_source):
        pattern = r'#\w+'
        return data_source.apply(lambda x: re.findall(pattern, x) if len(re.findall(pattern, x)) > 0 else np.nan)
    
    def extract_public_metrics(self, data_source, drop_source=True):
        data_source = data_source.reset_index(drop=True)
        public_metrics_2 = pd.DataFrame(data_source['public_metrics'].tolist())
        data_source = pd.concat([data_source, public_metrics_2], axis=1)

        if drop_source:
            data_source.drop('public_metrics', axis=1, inplace=True)

        return data_source
