#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
import matplotlib.pyplot as plt
import datetime
from nltk.tokenize import word_tokenize

from string import punctuation
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.lancaster import *
import os


MAIN_PATH = os.getcwd()
DATA_PATH = 'data'
MODEL_PATH = 'models'
RESULTS_PATH = 'results'
ENV_PATH = os.path.join(MAIN_PATH, '.env')
SLANG_PATH = os.path.join(MAIN_PATH, DATA_PATH, 'slang-dict.txt')

MATCH_DATA=pd.DataFrame({
            'match':['TOT-LIV','PSG-LIV','MAN-NEW','PSG-LYO'],
            'file_name':['TOT_LIV.csv','PSG_LIV.csv','MAN_NEW.csv','PSG_LYO.csv'],
            'team1':['Tottenham','PSG','Manchester','PSG'],
            'team2':['Liverpool','Liverpool','Newcastle','Lyon'],
            'match_start':['2018-09-15 11:30:00','2018-09-18 19:00:00','2018-10-06 16:30:00','2018-10-07 19:00:00'],
            'first_part_end':['2018-09-15 12:17','2018-09-18 19:46', '2018-10-06 17:17:00', '2018-10-07 19:49:00'],
            'second_part_start':['2018-09-15 12:32','2018-09-18 20:01','2018-10-06 17:32:00','2018-10-07 20:04:00'],
            'match_end':['2018-09-15 13:23','2018-09-18 20:55','2018-10-06 18:17:00','2018-10-07 20:49:00'],
                'hashtags_team1':[['#tottenham','#thfc','#spurs',"#tottenhamhotspur"],['#psg','#parissaintgermain'],['#mufc','#manutd','#manchesterunited','#reddevils','#manunited','#redarmy','#mufcfans'],['#psg','#parissaintgermain','#psgol']],
            'hashtags_team2':[['#liverpoolfc','#liverpool','#lfc'],['#liverpoolfc','#liverpool','#lfc'],['#newcastle','#newcastleunited','#nufc'],['#lyon','#teamol','#ol','#olympiclyon', '#olympiquelyonnais','#lyonfrance','#olympic']]})


def load_match_data(match):    
    file_name = MATCH_DATA.loc[MATCH_DATA['match']==match,'file_name'].iloc[0]
    team1 = MATCH_DATA.loc[MATCH_DATA['match']==match,'team1'].iloc[0]
    team2 = MATCH_DATA.loc[MATCH_DATA['match']==match,'team2'].iloc[0]
    match_start= MATCH_DATA.loc[MATCH_DATA['match']==match,'match_start'].iloc[0]
    first_part_end = MATCH_DATA.loc[MATCH_DATA['match']==match,'first_part_end'].iloc[0]
    second_part_start = MATCH_DATA.loc[MATCH_DATA['match']==match,'second_part_start'].iloc[0]
    match_end = MATCH_DATA.loc[MATCH_DATA['match']==match,'match_end'].iloc[0]
    hashtags_team1 = MATCH_DATA.loc[MATCH_DATA['match']==match,'hashtags_team1'].iloc[0]
    hashtags_team2 = MATCH_DATA.loc[MATCH_DATA['match']==match,'hashtags_team2'].iloc[0]
    return file_name, team1, team2, match_start, first_part_end, second_part_start, match_end, hashtags_team1, hashtags_team2

def create_match_df(match):
    file_name, team1, team2, match_start, first_part_end,\
        second_part_start, match_end, hashtags_team1, hashtags_team2 = load_match_data(match)
    file_path = os.path.join(MAIN_PATH, DATA_PATH, file_name)
    tweets_df = pd.read_csv(file_path,sep=';',encoding='utf-8', lineterminator='\n', index_col=0,parse_dates=['tweetCreated'])
    try:
        tweets_df['team\r']=tweets_df['team\r'].str.replace('\r','')
    except:
        pass
    tweets_df.columns=['tweetID', 'tweetText', 'tweetRetweetCt', 'tweetFavoriteCt',
       'tweetSource', 'tweetCreated', 'userID', 'userScreen', 'userName',
       'userCreateDt', 'userDesc', 'userFollowerCt', 'userFriendsCt',
       'userLocation', 'userTimezone', 'hashtag', 'team']
    tweets_df=tweets_df[((tweets_df['tweetCreated']>=match_start)&(tweets_df['tweetCreated']<=first_part_end))|((tweets_df['tweetCreated']>=second_part_start)&(tweets_df['tweetCreated']<=match_end))].reset_index(drop=True)
    return tweets_df


def load_slang(slang_path):
    slangdict = dict()
    with open(slang_path,'rt') as f:
        for line in f:
            spl = line.split('\t')
            slangdict[spl[0]] = spl[1][:-1]
    return slangdict


def processTweet(tweet, translate_slang=True, SLANG_DICT=load_slang(SLANG_PATH)):
    CONTRACTION_DICT = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "i'd": "i would","ive":"i have", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "one's":"", "someone's":"", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is","shoulda":"should have", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }

    URL_PATTERN=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    HASHTAG_PATTERN = re.compile(r'#([^\s]+)') #re.sub(HASHTAG_PATTERN, r'\1', '#abc')
    MENTION_PATTERN = re.compile(r'@[^\s]+')
    HTML_ENT_PATTERN=re.compile(r'\&\w*;')
    GOOD_PATTERN = re.compile(r'(\s)(>+)') #re.sub(GOOD_PATTERN, r'\1good', text)
    BAD_PATTERN = re.compile(r'(\s)(<+)')
    NEW_LINE_CHAR_PATTERN = re.compile(r'\\n|\\r|\\r\\n')
    WHITE_SPACE = re.compile(r'\s+')
   
    DUPLICATE_LETTERS_PATTERN = re.compile(r'([a-zA-Z])\1+')
    LEAGUE_EXPR=['premier league','champions league'] #re.sub(pat,r'\1\1','goooooaaaaaalll')
    
    
   
    tweet = tweet.lower()
   
    tweet = re.sub(URL_PATTERN, '', tweet)
    tweet = re.sub(HASHTAG_PATTERN, '', tweet)
    tweet = re.sub(MENTION_PATTERN, '', tweet)
    tweet = re.sub(HTML_ENT_PATTERN,'',tweet)
    tweet = re.sub(NEW_LINE_CHAR_PATTERN,'',tweet)
    tweet = re.sub(GOOD_PATTERN,r'\1good',tweet)
    tweet = re.sub(BAD_PATTERN,r'\1bad',tweet)
    tweet = re.sub(DUPLICATE_LETTERS_PATTERN,r'\1\1',tweet)
    tweet = re.sub(r'\d','',tweet)
    tweet = re.sub(r'(?!\.)(\.\.)(?!\.)',r'\1',tweet)
    tweet = re.sub(WHITE_SPACE,' ',tweet)
    
    
        
    for key,item in CONTRACTION_DICT.items():
        tweet = re.sub(key+'\s', item+' ', tweet)
    table = str.maketrans({key: None for key in punctuation})
    tweet = tweet.translate(table)
    
    translated_tweet_list = []
    if translate_slang:
        for token in word_tokenize(tweet):
            for key,item in SLANG_DICT.items():
                if token == key:
                    token = token.replace(key,item)        
            translated_tweet_list.append(token)
        tweet = ' '.join(translated_tweet_list)
    
    for item in LEAGUE_EXPR:
        tweet = re.sub(item, r'', tweet)
    
    tweet = tweet.strip().encode('ascii', 'ignore').decode('utf-8')
    tweet = re.sub(r'"', r'', tweet)
    
    
    return tweet


def tokenizeTweet(tweet, remove_stopwords = True, remove_shortwords = True, token_len = 2):
    STOP_WORDS = []
    if remove_stopwords:
        STOP_WORDS = stopwords.words('english')
    NEUTRAL_HASHTAHS = ['v','vs','league','pl','link','live','stream','hd', 'tottenham','thfc','spurs', 'liverpoolfc','liverpool','lfc', 'ynwa','fpl','matchday','coyg', 'epl', 'mufc','game','iphone','android','pc','mobile','ipad']
    ADDITIONAL_STOP_WORDS = ['amp', 'rt', 'via', 'fav',"'s"]
    ALL_STOP_WORDS=STOP_WORDS+NEUTRAL_HASHTAHS+ADDITIONAL_STOP_WORDS
        
    tokens = word_tokenize(tweet)
    tokens = [token.strip() for token in tokens if token not in ALL_STOP_WORDS]
    if remove_shortwords:
        tokens = [token for token in tokens if len(token) > token_len]
    return tokens

def stemTweet(tweet_tokenized):
    stemmer = LancasterStemmer()
    return [stemmer.stem(i) for i in tweet_tokenized]

def join_tokens(tweet_tokenized):
    return ' '.join(tweet_tokenized)

def tweets_to_text(tweet_series):
    return ' '.join(tweet for tweet in tweet_series)

def normalize_desc(s):
    return re.sub("[^a-zA-Z0-9']", ' ', s).lower().strip()

def get_processed_tweets(tweets_df, slang_dict, remove_stopwords, remove_shortwords, stemmed, labels=False):
    tweets_df['text'] = tweets_df['tweetText'].apply(lambda x: processTweet(x, slang_dict))
    tweets_df['text_tokens'] = tweets_df['text'].apply(lambda x: tokenizeTweet(x, remove_stopwords, remove_shortwords))
    tweets_df['tweet_stemmed'] = tweets_df['text_tokens'].apply(lambda x: stemTweet(x))
    tweets_df['tidy_tweet'] = tweets_df['text_tokens'].apply(join_tokens)
    tweets_df['tidy_tweet_stemmed'] = tweets_df['tweet_stemmed'].apply(join_tokens)
    if stemmed:
        corpus = tweets_df['tidy_tweet_stemmed']
    else:
        corpus = tweets_df['tidy_tweet']
    X = list(corpus)
    if labels:
        y = list(tweets_df['sentiment'])
        return X,y
    else:
        return X


# In[ ]:




