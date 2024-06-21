def process_predict(df):
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
    nltk.download('stopwords')
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
    return df
