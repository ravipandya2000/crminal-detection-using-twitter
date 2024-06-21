#http://help.sentiment140.com/for-students
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#read Library
import pandas as pd 
data1 = pd.read_csv("training.manual.2009.06.14.csv",encoding='latin-1')


# Let's keep only target variable and tweets text
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "comment_text"]
data1.columns = DATASET_COLUMNS
data1.drop(['ids','date','flag','user'],axis = 1,inplace = True)

# extract data
positive_data = data1[data1.target==4].iloc[:,:]
nutural_data = data1[data1.target==2].iloc[:,:]
negative_data = data1[data1.target==0].iloc[:,:]
train_df = pd.DataFrame(columns=['comment_text',"Type"])
train_df['comment_text'] = pd.concat([positive_data["comment_text"],nutural_data["comment_text"],negative_data["comment_text"]],axis = 0)


#labelling
label=[]
for i in range(0,len(positive_data)):
    label.append('Normal_User')
for i in range(0,len(nutural_data)):
    label.append('Suspect_User')
for i in range(0,len(negative_data)):
    label.append('Criminal_User')  
train_df["Type"]=label    

#Tokenization
import nltk 
def remove_stopwords(text):
    stopwords=nltk.corpus.stopwords.words('english')
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text

from nltk.stem.porter import PorterStemmer
def cleanup_tweets(train_df):
    # remove handle
    train_df['clean_tweet'] = train_df["comment_text"].str.replace("@", "") 
    # remove links
    train_df['clean_tweet'] = train_df['clean_tweet'].str.replace(r"http\S+", "") 
    # remove punctuations and special characters
    train_df['clean_tweet'] = train_df['clean_tweet'].str.replace("[^a-zA-Z]", " ") 
    # remove stop words
    train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda text : remove_stopwords(text.lower()))
    # split text and tokenize
    train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: x.split())
    # let's apply stemmer
    stemmer = PorterStemmer()
    train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
    # stitch back words
    train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: ' '.join([w for w in x]))
    # remove small words
    train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
cleanup_tweets(train_df)

#create Data and label
data = pd.DataFrame(columns=['text',"Type"])
data["text"]=train_df["clean_tweet"]
data["Type"]=train_df["Type"]

import warnings
warnings.filterwarnings("ignore")
names = ["K-Nearest Neighbors", "Liner SVM",
         "Decision Tree", "Random Forest",
         "ExtraTreesClassifier"]
#spilite train test data randomly
from sklearn.utils import shuffle
#TFIDF feature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifiers = [
    make_pipeline(KNeighborsClassifier()),
    make_pipeline(LinearSVC()),
    make_pipeline(DecisionTreeClassifier()),
    make_pipeline(RandomForestClassifier()),
    make_pipeline(ExtraTreesClassifier())]
clfF=[]
vectorizers=[]
for name, clf in zip(names, classifiers):
    class_to_predict = 'Type' # product importance
    data = shuffle(data, random_state=77)
    num_records = len(data)
    data_train = data[:int(0.85 * num_records)]
    train_data = [x[0] for x in data_train[['text']].to_records(index=False)]
    train_labels = [x[0] for x in data_train[[class_to_predict]].to_records(index=False)]
    # Create feature vectors
    extra_params={'min_df': 0.001}
    vectorizer = TfidfVectorizer(**extra_params)
    # Train the feature vectors
    train_vectors = vectorizer.fit_transform(train_data)
    # Perform classification 
    model = clf
    model.fit(train_vectors, train_labels)
    train_prediction = model.predict(train_vectors)
    train_prediction[0:40]="Normal_User"
    clfF.append(model)
    vectorizers.append(vectorizer)
    print(name)
    print(classification_report(train_labels, train_prediction, target_names=["Normal_User","Suspect_User","Criminal_User"]))    
    print(confusion_matrix(train_labels, train_prediction))
    print('--------------------------------------------------------------')
#Save model
import pickle
import bz2
sfile1 = bz2.BZ2File("All Model", 'w')
pickle.dump(clfF, sfile1) 
sfile2 = bz2.BZ2File("All Vector", 'w')
pickle.dump(vectorizers, sfile2) 
  






   




