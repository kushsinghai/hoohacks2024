import os
import pandas as pd
from sklearn.model_selection import train_test_split

full_df = pd.DataFrame()

def create_df():
    base_directory_path = 'Data'  # base directory containing artist folders
    data = []  # list holding texts and labels

    # iterate through each artist directory within data directory
    for artist_name in os.listdir(base_directory_path):
        artist_directory_path = os.path.join(base_directory_path, artist_name)
        
        # check if dir
        if os.path.isdir(artist_directory_path):
            # iterate through each file in the artist directory
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

def split_test_train(full_df):
    # Split the data into features (X) and labels (y)
    X = full_df['lyrics']  # Features (the lyrics)
    y = full_df['artist']  # Labels (the artist names)

    # Split the dataset into training and testing sets
    # test_size specifies the proportion of the dataset to include in the test split
    # random_state is a seed value for reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Now you have your data split into:
    # X_train, y_train: training features and labels
    # X_test, y_test: testing features and labels

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

def main():
    create_df()
    split_test_train(full_df)

main()