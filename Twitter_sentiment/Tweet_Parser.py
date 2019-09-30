
# coding: utf-8

# In[1]:

import json
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import gensim
import spacy
import pyspark
from pyspark import *


# In[79]:

tweets_data = []
#tweets_file = open(tweets_data_path, "r")

with open('/Users/uun466/Desktop/Data-Science-Project/tweet_file.json') as json_data:
    d = json.load(json_data)
    if len(d) > 1:
        for each in range(len(d)):
            tweets_data.append(d[each])
    else:
        tweets_data.append(d)


# In[81]:

print len(tweets_data)


# In[80]:

tweets_data


# In[84]:

tweets = pd.DataFrame()


# In[86]:

tweets['text'] = map(lambda tweet: tweet['text'].strip(), tweets_data)
tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)
tweets['user_nm'] = map(lambda tweet: tweet['user']['name'].encode('utf-8'), tweets_data)
tweets['coordinates'] = map(lambda tweet: tweet['coordinates'], tweets_data)
tweets['location'] = map(lambda tweet: tweet['user']['location'], tweets_data)
tweets['retweets_count'] = map(lambda tweet: tweet['retweet_count'], tweets_data)
#tweets['retweet_location'] = map(lambda tweet: tweet['retweeted_status']['user']['location'], tweets_data)


# In[87]:

tweets['retweets_count']


# In[ ]:

get_ipython().magic(u'matplotlib inline')


# In[88]:

tweets


# In[89]:

## Get Verbatim text
verbatim = [str(v.encode('utf-8')) for v in tweets.text.values.tolist()]


# In[90]:

verbatim


# In[91]:

import textauger
import re


# In[92]:

from textauger import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Vader
from textauger import textfeatures


# In[93]:

tweets['text_clean'] = [re.sub(r"http\S+", "", v) for v in tweets.text.values.tolist()]
tweets['text_clean'] = [re.sub(r"#\S+", "", v) for v in tweets.text_clean.values.tolist()]
tweets['text_clean'] = [re.sub(r"@\S+", "", v) for v in tweets.text_clean.values.tolist()]
tweets['text_clean'] = [re.sub(r"u'RT\S+", "", v) for v in tweets.text_clean.values.tolist()]
tweets['text'] = [v.replace('\n'," ") for v in tweets.text.values.tolist()]


# In[94]:

tweets['text_clean'] = preprocessing.clean_text(text=tweets.text_clean.values,
                         remove_short_tokens_flag=False,
                         lemmatize_flag=True)


# In[ ]:

tweets['text_clean'][1]


# In[95]:

tweets['sentiment_score'] = [textfeatures.score_sentiment(v)['compound'] for v in tweets.text_clean.values.tolist()]


# In[96]:

tweets


# In[ ]:

textfeatures.score_sentiment(tweets['text_clean'][1])


# In[97]:

tweets.loc[tweets['sentiment_score'] > 0.0, 'sentiment'] = 'positive'
tweets.loc[tweets['sentiment_score'] == 0.0, 'sentiment'] = 'neutral'
tweets.loc[tweets['sentiment_score'] < 0.0, 'sentiment'] = 'negative'


# In[98]:

tweets


# In[99]:

tweets.to_csv("/Users/uun466/Desktop/Data-Science-Project/tweet_file.csv", encoding = 'utf-8')


# In[100]:

df = sqlContext.read.load('/Users/uun466/Desktop/Data-Science-Project/tweet_file.csv',
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')


# In[101]:

df.show()


# In[ ]:




# In[ ]:
