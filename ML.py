import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
import joblib

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

# vectorize converts text into algorithmic format
def vectorize(X_train_lyrics, X_test_lyrics):
    # Temporarily remove the stop_words parameter or adjust the max_features
    tfidf_vectorizer = TfidfVectorizer()

    try:
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_lyrics)
    except ValueError as e:
        print(f"Error fitting TF-IDF on the training set: {e}")
        # Inspect the training data to understand the issue better
        print(X_train_lyrics)
        raise

    X_test_tfidf = tfidf_vectorizer.transform(X_test_lyrics)
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

# initialize multinomial naive bayes classifier
def train_naive_bayes(X_train_tfidf, y_train):
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, X_test_tfidf, y_test):
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions) 
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

def main():
    df = create_df()
    X_train, X_test, y_train, y_test = split_test_train()
    # Only pass the lyrics for TF-IDF vectorization
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize(X_train['lyrics'], X_test['lyrics'])

    # Make sure to include the WPM feature after TF-IDF vectorization
    # Stack the WPM feature onto the TF-IDF feature matrix
    X_train_tfidf = sp.hstack((X_train_tfidf, np.array(X_train['wpm']).reshape(-1, 1)), format='csr')
    X_test_tfidf = sp.hstack((X_test_tfidf, np.array(X_test['wpm']).reshape(-1, 1)), format='csr')
    print(X_test_tfidf)
    #imputer = SimpleImputer(strategy='mean')

    #X_train_tfidf = imputer.fit_transform(X_train_tfidf)
    #X_test_tfidf = imputer.transform(X_test_tfidf)
    model = train_naive_bayes(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)
    joblib.dump(model, 'naive_bayes_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

main()