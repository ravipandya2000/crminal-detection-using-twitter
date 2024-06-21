import streamlit as st
import pandas as pd
import hashlib
from PIL import Image
import pickle
import bz2
import numpy as np
import json
from process_predict import process_predict
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
# Instantiate an object
python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

def make_hashes(password):   
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(FirstName TEXT,LastName TEXT,Mobile TEXT,Email TEXT,password TEXT,Cpassword TEXT)')
def add_userdata(FirstName,LastName,Mobile,Email,password,Cpassword):
    c.execute('INSERT INTO userstable(FirstName,LastName,Mobile,Email,password,Cpassword) VALUES (?,?,?,?,?,?)',(FirstName,LastName,Mobile,Email,password,Cpassword))
    conn.commit()
def login_user(Email,password):
    c.execute('SELECT * FROM userstable WHERE Email =? AND password = ?',(Email,password))
    data = c.fetchall()
    return data
def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data



def main():
    st.title("Welcome To Crime User Prediction")
    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        original_title="<p style='text-align: center;'>Twitter is used extensively in the United States as well as globally, creating many opportunities to augment decision support systems with Twitter-driven predictive analytics. Twitter is an ideal data source for decision support: its users, who number in the millions, publicly discuss events, emotions, and innumerable other topics; its content is authored and distributed in real time at no charge; and individual messages (also known as tweets) are often tagged with precise spatial and temporal coordinates. This article presents research investigating the use of spatiotemporally tagged tweets for crime prediction.</p>"
        image = Image.open('flow.jpg')
        st.image(image)
        st.markdown(original_title, unsafe_allow_html=True)
    elif choice == "Login":
        st.subheader("Login Section")
        Email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(Email,check_hashes(password,hashed_pswd))
            if result:
                location=['Gujarat','UP','Maharastra','Delhi']
                choice = st.selectbox("Select Location",location)
                if choice=='Gujarat':
                    geocode = '28.6517178,77.2219388,1000mi' # latitude,longitude,distance(mi/km)
                if choice=='UP':
                    geocode = '28.6517178,77.2219388,1000mi' # latitude,longitude,distance(mi/km)
                if choice=='Maharastra':
                    geocode = '28.6517178,77.2219388,1000mi' # latitude,longitude,distance(mi/km)
                if choice=='Delhi':
                    geocode = '28.6517178,77.2219388,1000mi' # latitude,longitude,distance(mi/km)
                texts=str(st.text_input("Enter Keyword with AND and OR operator"))
                keywords=texts+" -filter:retweets"
                query = {'q': keywords,
                        'count': 100,
                        'lang': 'en',
                        'geocode': geocode,
                        }
                if st.button('Retrive Tweets'):
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
                    st.dataframe(df)
                    df1=process_predict(df)
                if st.button("Process and Predict"):
                    import pandas as pd
                    df1=pd.read_csv("tweets.csv")
                    st.dataframe(df1)   
            else:
                st.warning("Incorrect Email/Password")
                
    elif choice == "SignUp":
        FirstName = st.text_input("Firstname")
        LastName = st.text_input("Lastname")
        Mobile = st.text_input("Mobile")
        Email = st.text_input("Email")
        new_password = st.text_input("Password",type='password')
        Cpassword = st.text_input("Confirm Password",type='password')
        if st.button("Signup"):
            create_usertable()
            add_userdata(FirstName,LastName,Mobile,Email,make_hashes(new_password),make_hashes(Cpassword))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
           
if __name__ == '__main__':
    main()