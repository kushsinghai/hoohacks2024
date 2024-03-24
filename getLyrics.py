import os
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# define list of songs + artist
songs = ["Be alright", "What Do You Mean", "Sorry", "Hold On", "Love Yourself"]
songs += ["Yummy", "Boyfriend", "Off My Face", "Anyone", "Company"]
songs += ["Hailey", "Somebody To Love", "One Time", "All That Matters", "Come Around Me"]
songs += ["U smile", "2 much", "Lifetime", "Purpose", "Habitual"]
artist = "Justin Bieber"

# director to save cleaned data
parent_dir = "Data"
save_dir = os.path.join(parent_dir, artist.replace(" ", "_").lower())
os.makedirs(save_dir, exist_ok=True)
global word_count

def fetch_lyrics(artist, title):
    """Fetch lyrics for a given song."""
    base_url = "https://api.lyrics.ovh/v1"
    response = requests.get(f"{base_url}/{artist}/{title}")
    if response.status_code == 200:
        return response.json().get('lyrics', '')
    return None

# def count_words_in_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         contents = file.read()
#         words = contents.split()
#         word_count = len(words)
#     return word_count

def clean_lyrics(lyrics, artist):
    """Clean lyrics by removing stopwords, punctuation, making lowercase, and removing 'paroles de chanson'."""
    stop_words = set(stopwords.words('english'))
    # remove punctuation + make lower case
    lyrics = re.sub(r'[^\w\s]', '', lyrics).lower()
    lyrics = lyrics.replace('paroles de la chanson', '')
    artist_pattern = re.escape('par ' + artist.lower())
    lyrics = re.sub(artist_pattern, '', lyrics)
    # tokenize + remove stopwords
    tokenized = word_tokenize(lyrics)
    # cleaned = [word for word in tokenized if word not in stop_words]
    cleaned = [word for word in tokenized]
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
        # count_words_in_file(song)
        cleaned_lyrics = clean_lyrics(raw_lyrics, artist)
        save_cleaned_lyrics(song, cleaned_lyrics)
        print(f"Processed and saved lyrics for '{song}'")
    else:
        print(f"Lyrics for '{song}' not found or failed to fetch.")

print("All done!")
