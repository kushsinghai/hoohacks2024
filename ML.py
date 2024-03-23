# ML.py

import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the set of stopwords from NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def clean_lyrics(lyrics):
    # Convert to lowercase
    lyrics = lyrics.lower()
    
    # Remove punctuation and numbers
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Removes all characters except word characters and whitespace
    lyrics = re.sub(r'\d+', '', lyrics)  # Removes numbers
    
    # Tokenize the lyrics into words
    words = word_tokenize(lyrics)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join words back into a single string
    cleaned_lyrics = ' '.join(filtered_words)
    
    return cleaned_lyrics

# Directory containing the song files
song_dir = 'Kendrick'

# Loop through all files in the directory
for filename in os.listdir(song_dir):
    if filename.endswith('.txt'):  # Check if the file is a text file
        file_path = os.path.join(song_dir, filename)
        
        # Read the original lyrics
        with open(file_path, 'r') as file:
            original_lyrics = file.read()
        
        # Clean the lyrics
        cleaned_lyrics = clean_lyrics(original_lyrics)
        
        # Define a new filename for the cleaned lyrics
        cleaned_filename = 'cleaned_' + filename
        cleaned_file_path = os.path.join(song_dir, cleaned_filename)
        
        # Write the cleaned lyrics to a new file
        with open(cleaned_file_path, 'w') as file:
            file.write(cleaned_lyrics)
            print(f"Cleaned lyrics have been written to '{cleaned_file_path}'")
