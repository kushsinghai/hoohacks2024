import os
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define the list of songs and the artist
songs = ["The Search", "Time", "Lie"]  # Add more songs as needed
artist = "NF"

# Directory to save cleaned data
save_dir = artist.replace(" ", "_").lower()
os.makedirs(save_dir, exist_ok=True)

def fetch_lyrics(artist, title):
    """Fetch lyrics for a given song."""
    base_url = "https://api.lyrics.ovh/v1"
    response = requests.get(f"{base_url}/{artist}/{title}")
    if response.status_code == 200:
        return response.json().get('lyrics', '')
    return None

def clean_lyrics(lyrics):
    """Clean lyrics by removing stopwords, punctuation, making lowercase, and removing 'paroles de chanson'."""
    stop_words = set(stopwords.words('english'))
    # Remove punctuation and make lowercase
    lyrics = re.sub(r'[^\w\s]', '', lyrics).lower()
    lyrics = lyrics.replace('paroles de la chanson', '')
    # Tokenize and remove stopwords
    tokenized = word_tokenize(lyrics)
    cleaned = [word for word in tokenized if word not in stop_words]
    return ' '.join(cleaned)

def save_cleaned_lyrics(title, lyrics):
    """Save cleaned lyrics to a text file within the artist's directory."""
    filename = f"{title.lower().replace(' ', '_')}.txt"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(lyrics)

for song in songs:
    raw_lyrics = fetch_lyrics(artist, song)
    if raw_lyrics:
        cleaned_lyrics = clean_lyrics(raw_lyrics)
        save_cleaned_lyrics(song, cleaned_lyrics)
        print(f"Processed and saved lyrics for '{song}'")
    else:
        print(f"Lyrics for '{song}' not found or failed to fetch.")

print("All done!")
