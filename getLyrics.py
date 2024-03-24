import os
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from getSongDuration import getDuration

# define list of songs + artist
songs = ["Shape of You", "Galway Girl", "Happier", "Eraser", "Barcelona"]
songs += ["Sing", "New Man", "Photograph", "Tenerife Sea", "Thinking out Loud"]
songs += ["Shivers", "Bad Habits", "Save Myself", "Overpass Graffiti", "The A Team"]
songs += ["Lego House", "Give Me Love", "Kiss Me", "Tides", "First Times"]
songs += ["Perfect", "Eyes Closed", "End of Youth", "Castle on the Hill", "Celestial"]
songs += ["Curtains", "American Town", "Tenerife Sea", "Boat", "Drunk"]
artist = "Ed Sheeran"


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
    cleaned = [word for word in tokenized if word not in stop_words]
    # Check that there are lyrics left after cleaning
    if not cleaned:
        raise ValueError(f"After cleaning, no lyrics remain for the song by {artist}. Check the cleaning process.")
    return ' '.join(cleaned)

def save_cleaned_lyrics(title, lyrics):
    """Save cleaned lyrics to a text file within the artist's directory."""
    global word_count
    filename = f"{title.lower().replace(' ', '_')}.txt"
    filepath = os.path.join(save_dir, filename)
    word_count = len(lyrics.split())
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(lyrics + "\n")
        file.write(str(word_count / getDuration(artist, title)))

    
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
