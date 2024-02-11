import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
import re

# Load the model from disk
loaded_model = pickle.load(open('Assignment4(1).pkl', 'rb'))

# 1 Setup your environment (pandas, scikit learn, numpy, nltk)
!pip install nltk -q

#clean the data, remove extra characters, punct, symbols, stemming, lemmatize the information, tokenization, TF-IDF, BoW

from tkinter import Text
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer # takes text and turns it into numbers
from sklearn.metrics.pairwise import linear_kernel  # takes vectors (in our case that represents text) and computes how "close" they are
import nltk
from nltk import download
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
from io import StringIO
import re
import nltk

# Initialize the lemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Function to clean and lemmatize text
def clean_and_lemmatize(text):
    # Remove non-alphabetic characters and split into words
    words = re.sub(r"[^a-zA-Z]", " ", text).split()
    # Lemmatize each word
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in words])

# Download the Tweets.csv data
text = '/content/Car details v3.csv'
df = pd.read_csv(text)

data = pd.DataFrame(df)
# Display the first few rows of the DataFrame
data.head()

# Preprocess the text data

def clean_text(data):
  stop_words = stopwords.words('english')
  data = text.lower()
  data = ''.join([char for char in data if char not in string.punctuation])
  data_filtered = [word for word in data if word.lower() not in stop_words]
  return data

data.apply(lambda x:clean_text(x))
data.head()

# 3 Stem and Lemma

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function to apply stemming
def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Function to apply lemmatization
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Applying stemming and lemmatization
df['stemmed_text'] = data['cleaned_text'].apply(lambda x: stem_text(x))
df['lemmatized_text'] = data['cleaned_text'].apply(lambda x: lemmatize_text(x))

#4 TF-IDF, BoW and the ML Model
#logistic regression will be used for this model with binary classification


from sklearn.model_selection import train_test_split

#split our data

X_train, X_test, y_train, y_test = train_test_split(data['name'], df['transmission'], test_size=0.25, random_state = 21)

# Creating a model with VECTORIZED DATA

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Creating a pipeline with TF/IDF vectorizer and Logistic Regression
model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])

# Training the model
model.fit(X_train, y_train)

# Evaluation my NLP Model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Precision:", precision_score(y_test, y_pred))
#print("Recall:", recall_score(y_test, y_pred))
#print("F1-Score:", f1_score(y_test, y_pred))

import pickle
filename = 'Assignment4.pkl'
pickle.dump(model,open(filename, 'wb'))
# Set up the Streamlit interface
st.title('Sentiment Analysis Model')
user_input = st.text_area("Enter Text", "Type Here...")
if st.button('Predict Sentiment'):
    result = predict_sentiment(user_input)
    if result == 1:
        st.success('The sentiment is positive!')
    else:
        st.error('The sentiment is negative or neutral.')

# To run the app, use the following command:
# streamlit run app.py
