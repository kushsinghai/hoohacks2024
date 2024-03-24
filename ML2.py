import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load your dataset
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
                        wpm_line = lines[1].strip()  # Extract the WPM from the second line
                        # Ensure that the second line is indeed a float
                        try:
                            wpm = float(wpm_line)
                        except ValueError:
                            # Handle the case where conversion to float fails
                            print(f"Could not convert WPM value to float in file: {filename}")
                            wpm = None
                        lyrics = ''.join(lines[2:])  # Join the rest of the lines to form the lyrics
                        data.append({'Lyrics': lyrics, 'Artist': artist_name, 'WPM': wpm})  # Changed to correct case
                    else:
                        print(f"File {filename} does not contain enough lines for WPM information.")

df = pd.DataFrame(data)

# Make sure the dataframe column names are correct
print(df.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Lyrics', 'WPM']], df['Artist'], test_size=0.2, random_state=42)

# Use 'Lyrics' column to vectorize the text data. 
vectorizer = TfidfVectorizer()
X_train_lyrics = vectorizer.fit_transform(X_train['Lyrics'])
X_test_lyrics = vectorizer.transform(X_test['Lyrics'])

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_lyrics, y_train)

# Evaluate the model
predictions = model.predict(X_test_lyrics)
print(classification_report(y_test, predictions))
