import os
import re
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
import joblib
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize





full_df = pd.DataFrame()

# creating the data frame since all data is in folders and across files
def create_df():
    base_directory_path = 'Data'
    data = []  # list holding texts, labels, and wpm
    # iterate through each artist directory within data directory
    for artist_name in os.listdir(base_directory_path):
        artist_directory_path = os.path.join(base_directory_path, artist_name)
        if os.path.isdir(artist_directory_path):  # check if dir & iterate through each file in the artist directory
            for filename in os.listdir(artist_directory_path):
                if filename.endswith('.txt'):  # check for .txt
                    file_path = os.path.join(artist_directory_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        if len(lines) > 1:
                            lyrics = lines[0].strip()
                            wpm_line = lines[1].strip()  # Extract the WPM from the second line
                            # Ensure that the second line is indeed a float
                            try:
                                wpm = float(wpm_line)
                            except ValueError:
                                # Handle the case where conversion to float fails
                                print(f"Could not convert WPM value to float in file: {filename}")
                                wpm = None
                            #lyrics = ''.join(lines[2:])  # Join the rest of the lines to form the lyrics
                            data.append({'lyrics': lyrics, 'artist': artist_name, 'wpm': wpm})
                        else:
                            print(f"File {filename} does not contain enough lines for WPM information.")
    global full_df
    full_df = pd.DataFrame(data)
    

# split the data into features (X) and labels (y)
def split_test_train():
    global full_df, wpm
    print(full_df)
    X = full_df[['lyrics', 'wpm']]
    y = full_df['artist'] 
    
    # split data into train and test
    # test_size is the proportion of the dataset to include in test split
    # random_state is a seed value for reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test

def vectorize(X_train_lyrics, X_train_wpm, X_test_lyrics, X_test_wpm):
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    # Fit the model and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_lyrics)

    # Transform the test data
    X_test_tfidf = tfidf_vectorizer.transform(X_test_lyrics)
    
    # Handle the 'wpm' feature. Assuming it's a dense array, no need for imputation as there are no NaNs based on the error screenshot
    X_train_wpm = np.array(X_train_wpm).reshape(-1, 1)
    X_test_wpm = np.array(X_test_wpm).reshape(-1, 1)

    # Stack the WPM feature onto the TF-IDF feature matrix
    X_train_tfidf = sp.hstack((X_train_tfidf, X_train_wpm), format='csr')
    X_test_tfidf = sp.hstack((X_test_tfidf, X_test_wpm), format='csr')

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer



# initialize multinomial naive bayes classifier
def train_naive_bayes(X_train_tfidf, y_train):
    # Using GridSearchCV to find the best alpha parameter
    nb_model = MultinomialNB()
    parameters = {'alpha': [0.01, 0.1, 1, 10, 100]}
    clf = GridSearchCV(nb_model, parameters, cv=5)
    clf.fit(X_train_tfidf, y_train)
    
    return clf.best_estimator_
def evaluate_model(model, X_test_tfidf, y_test):
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions) 
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

def predict_artist(input_lyrics, input_wpm):


    # Load the vectorizer and the model
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('naive_bayes_model.pkl')

    # Transform the lyrics with the vectorizer
    lyrics_tfidf = tfidf_vectorizer.transform([input_lyrics])

    # Stack the WPM feature onto the TF-IDF features
    wpm_feature = np.array([input_wpm]).reshape(-1, 1)
    combined_features = sp.hstack((lyrics_tfidf, wpm_feature))

    # Predict the artist using the model
    predicted_artist = model.predict(combined_features)
    return predicted_artist[0]

def main():
    global full_df
    create_df()
    if not full_df.empty and 'wpm' in full_df.columns:
        X_train, X_test, y_train, y_test = split_test_train()
        X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize(
            X_train['lyrics'], X_train['wpm'],
            X_test['lyrics'], X_test['wpm']
        )
        model = train_naive_bayes(X_train_tfidf, y_train)
        evaluate_model(model, X_test_tfidf, y_test)
        joblib.dump(model, 'naive_bayes_model.pkl')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    else:
        print("The DataFrame is empty or does not contain a 'wpm' column.")

main()