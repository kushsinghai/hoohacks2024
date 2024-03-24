import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Authenticate with the Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id='69e14015c025495cb095a26340a954f5', client_secret='70f478fc63a545dfb1bb3dd66acb377f')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def getDuration(artist, songName):
    # search for the song
    query = 'track:{} artist:{}'.format(songName, artist)
    results = sp.search(q=query, type='track')
    track = results['tracks']['items'][0]

    # get duration of the track
    duration_ms = track['duration_ms']
    #print(duration_ms)
    return duration_ms / 60000


