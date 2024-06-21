import json
# Enter your keys/secrets as strings in the following fields
# authorization tokens
credentials = {}
credentials['CONSUMER_KEY'] = 'XV2kS4M1OmganL2zZU0q8Kyxh'
credentials['CONSUMER_SECRET'] = 'PvjekJXnI304fE2En3cmYuftP7yOXH0xiANsWOsW1nUpbwV4j7'
credentials['ACCESS_TOKEN'] = '152569292-Uw6KPJqudtctiYjpR1GEWOMYMKGc2DhczLiZq4Q4'
credentials['ACCESS_SECRET'] = 'Muv9NC0JhKiskMqt7hO7XNbCZPRBOAOtADNaAN8xeBQ1a'

# Save the credentials object to file
with open("twitter_credentials.json", "w") as file:
    json.dump(credentials, file)


from twython import Twython
import json

# Load credentials from json file
with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)
geocode = '28.6517178,77.2219388,1000mi' # latitude,longitude,distance(mi/km)
# Instantiate an object
python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
# Create our query
keywords="crime OR attempt -filter:retweets"
query = {'q': keywords,
        'count': 100,
        'lang': 'en',
        'geocode': geocode,
        }
import pandas as pd
# Search tweets
dict_ = {'user': [], 'date': [], 'text': [], 'user_loc': []}
for status in python_tweets.search(**query)['statuses']:
    dict_['user'].append(status['user']['screen_name'])
    dict_['date'].append(status['created_at'])
    dict_['text'].append(status['text'])
    dict_['user_loc'].append(status['user']['location'])
# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)

#emoticon feature
import demoji
import emoji
demoji.download_codes()
for i in range(len(df)):
    print(demoji.findall(df['text'][i]))
    df['text'][i]=emoji.demojize(df['text'][i], delimiters=("", ""))

#Pre-process
import string
import re
import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(tweet):
#     wnl = WordNetLemmatizer()
    cleaned_tweet = []

    words = tweet.split()
    for word in words:
        # Skip Hyperlinks and Twitter Handles @<user>
        if ('http' in word) or ('.com' in word) or ('www.' in word) or (word.startswith('@')):
            continue

        # Remove Digits and Special Characters
        temp = re.sub(f'[^{string.ascii_lowercase}]', '', word.lower())

        # Remove words with less than 3 characters
        if len(temp) < 3:
            continue

        # Store the Stemmed version of the word
        temp = stemmer.stem(temp)

        if len(temp) > 0:
            cleaned_tweet.append(temp)

    return ' '.join(cleaned_tweet)
cl=[]
for i in range(len(df)):
    cl.append(preprocess(df['text'][i]))
df['clean_tweet']=cl


#load model
import pickle
import bz2
sfile1 = bz2.BZ2File('All Model', 'r')
models=pickle.load(sfile1)
sfile2 = bz2.BZ2File('All Vector', 'r')
vectorizers=pickle.load(sfile2)

names = ["K-Nearest Neighbors", "Liner SVM",
         "Decision Tree", "Random Forest",
         "ExtraTreesClassifier"]
for i in range(0,len(names)):
    test_vectors = vectorizers[i].transform(cl)
    df['Class '+names[i]]=models[i].predict(test_vectors)
    df.to_csv('tweets.csv')
    
    
    
    
    
    
    
    
    
    