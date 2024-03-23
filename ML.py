import os
import pandas as pd

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
    full_df = pd.DataFrame(data)

def main():
    create_df()

main()