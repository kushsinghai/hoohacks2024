import os
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define the list of songs and the artist
songs = ["The Search", "Let You Down", "Lie", "HOPE", "HAPPY"]
songs += ["Paralyzed", "Lie", "Running", "When I Grow Up", "JUST LIKE YOU"]
songs += ["If You Want Love", "MISTAKE", "Hate Myself", "CLOUDS", "Remember This"]
songs += ["Time", "DRIFTING", "Oh Lord", "GONE", "Change"]
artist = "nf"

# Directory to save cleaned data
parent_dir = "Data"
save_dir = os.path.join(parent_dir, artist.replace(" ", "_").lower())
os.makedirs(save_dir, exist_ok=True)

def fetch_lyrics(artist, title):
    """Fetch lyrics for a given song."""
    base_url = "https://api.lyrics.ovh/v1"
    response = requests.get(f"{base_url}/{artist}/{title}")
    if response.status_code == 200:
        return response.json().get('lyrics', '')
    return None

def clean_lyrics(lyrics, artist):
    """Clean lyrics by removing stopwords, punctuation, making lowercase, and removing 'paroles de chanson'."""
    stop_words = set(stopwords.words('english'))
    # Remove punctuation and make lowercase
    lyrics = re.sub(r'[^\w\s]', '', lyrics).lower()
    lyrics = lyrics.replace('paroles de la chanson', '')
    artist_pattern = re.escape('par ' + artist.lower())
    lyrics = re.sub(artist_pattern, '', lyrics)
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
        cleaned_lyrics = clean_lyrics(raw_lyrics, artist)
        save_cleaned_lyrics(song, cleaned_lyrics)
        print(f"Processed and saved lyrics for '{song}'")
    else:
        print(f"Lyrics for '{song}' not found or failed to fetch.")

print("All done!")
