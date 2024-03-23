import requests

# List of twenty Drake songs
songs = [
    "Lover"
]

def fetch_lyrics(artist, title):
    """Fetch lyrics for a given song."""
    base_url = "https://api.lyrics.ovh/v1"
    response = requests.get(f"{base_url}/{artist}/{title}")
    if response.status_code == 200:
        return response.json().get('lyrics', 'Lyrics not found.')
    else:
        return "Failed to fetch lyrics."

def save_lyrics(title, lyrics):
    """Save lyrics to a text file."""
    filename = f"{title}.txt".replace("/", " or ")  # Replace any slashes in song titles
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(lyrics)

for song in songs:
    lyrics = fetch_lyrics("Taylor Swift", song)
    save_lyrics(song, lyrics)
    print(f"Saved lyrics for '{song}'")

print("All done!")
