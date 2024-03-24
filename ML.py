import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

full_df = pd.DataFrame()

# creating the data frame since all data is in folders and across files
def create_df():
    base_directory_path = 'Data'
    data = []  # list holding texts and labels
    # iterate through each artist directory within data directory
    for artist_name in os.listdir(base_directory_path):
        artist_directory_path = os.path.join(base_directory_path, artist_name)
        if os.path.isdir(artist_directory_path): # check if dir & iterate through each file in the artist directory
            for filename in os.listdir(artist_directory_path):
                if filename.endswith('.txt'):  # check for .txt
                    file_path = os.path.join(artist_directory_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lyrics = file.read()
                        data.append({'lyrics': lyrics, 'artist': artist_name})
    
    # convert list of df to directory
    global full_df
    full_df = pd.DataFrame(data)
    return full_df

# split the data into features (X) and labels (y)
def split_test_train(full_df):
    X = full_df['lyrics']  
    y = full_df['artist'] 

    # split data into train and test
    # test_size is the proportion of the dataset to include in test split
    # random_state is a seed value for reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test

# vectorize converts text into algorithmic format
def vectorize(X_train, X_test):
    # initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train) # fit model and transform the training data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
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
    full_df = create_df()
    X_train, X_test, y_train, y_test = split_test_train(full_df)
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize(X_train, X_test)
    model = train_naive_bayes(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)

main()