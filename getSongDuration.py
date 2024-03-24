import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Authenticate with the Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id='69e14015c025495cb095a26340a954f5', client_secret='70f478fc63a545dfb1bb3dd66acb377f')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def getDuration(artist, songName):
    # search for the song
    query = 'track:{} artist:{}'.format(songName, artist)
    results = sp.search(q=query, type='track')

    # Check if search results are present
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        # get duration of the track
        duration_ms = track['duration_ms']
        return duration_ms / 60000
    else:
        # Handle case where no results are found, perhaps return None or a default value
        print(f"No duration data found for '{songName}' by '{artist}'")
        return None

def getArtist(songName):
    query = 'track:{}'.format(songName)
    results = sp.search(q=query, type='track')
    if results['tracks']['items']:
        
        return results['tracks']['items'][0]['artists'][0]['name']
    else:
        # Handle case where no results are found, perhaps return None or a default value
        print(f"No duration data found for '{songName}' by '{artist}'")
        return None
    


def main():
    print(1048/getDuration("nf", 'Leave me alone'))
    print(getArtist('uptown funk'))
