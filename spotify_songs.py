import pandas as pd
import requests
import re
import time
import csv
import os
import sys

def get_tokens():
    """ Function to access updated access and refresh tokens from token.txt file """
    file_path = "token.txt"
    with open(file_path, 'r') as file:
        access = file.readline().strip().split("access: ")[1]
        refresh = file.readline().strip().split("refresh: ")[1]
    return access, refresh

access_token, refresh_token = get_tokens()
headers = {'Authorization': f'Bearer {access_token}'}
        
def search_artists(artist, headers):
    """ Function to search by artist name
        Returns the artist's spotify ID if found, otherwise returns None
        Inputs:
            - artist: artist name
    """
    url = 'https://api.spotify.com/v1/search'
    params = {'q': artist, 'type': 'artist'}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    artists = response.json()
    artist_ids = [item['id'] for item in artists['artists']['items']]
    for id in artist_ids:
        curr_artist = get_artist(id, headers)
        name = curr_artist["name"]
        if name == artist:
            popular = curr_artist["popularity"]
            if popular > 0:
                return id
    return None

def get_song(song, artist):
    params = {'q': f'artist:{artist} track:{song}', 'type': 'track'}
    for _ in range(3):
        response = requests.get('https://api.spotify.com/v1/search', params=params, headers=headers)
        try:
            response.raise_for_status()
            r = response.json()
            return r
        except:
            print(f"{response.status_code} Sleeping for 30 seconds ...")
            time.sleep(31)  
    print("Quitting ... ")
    sys.exit(0)

def get_audio_analysis(track_id, headers):
    """ Function to request audio analysis of a song
        Will make request up to 3 times with errors, then quits program
        Inputs:
            - track_id: the ID for the given song
    """
    for _ in range(3):
        response = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}', headers=headers)
        try: 
            response.raise_for_status()
            return response.json()
        except: 
            print(f"{response.status_code} Sleeping for 30 seconds ...")
            time.sleep(31)  
    print("Quitting ... ")
    sys.exit(0)

def get_album(album_id, headers):
    response = requests.get(f'https://api.spotify.com/v1/albums/{album_id}', headers=headers)
    response.raise_for_status()
    return response.json()

def get_artist(artist_id, headers):
    response = requests.get(f'https://api.spotify.com/v1/artists/{artist_id}', headers=headers)
    response.raise_for_status()
    return response.json()

def get_artist_top_song(id, headers):
    response = requests.get(f'https://api.spotify.com/v1/artists/{id}/top-tracks', params = {'market':'US'}, headers=headers)
    response.raise_for_status()
    response = response.json()
    if len(response["tracks"]) > 0:
        song, track_id = response["tracks"][0]["name"], response["tracks"][0]['id']
        return song, track_id
    else: return None, None

def get_show(id, headers):
    response = requests.get(f'https://api.spotify.com/v1/shows/{id}', params = {'market':'US'}, headers=headers)
    try:
        response.raise_for_status()
        response = response.json()
        return response
    except:
        return None

def search_podcasts(csv_file):
    dict = {"Podcast Name":[], "Podcast ID":[], "Episode Name":[], "Description":[]}
    file=open(os.getcwd() + csv_file, "r")
    reader = csv.reader(file)
    l=0
    for line in reader:
        id, name = line[2], line[3]
        if l > 0:
            show = get_show(id, headers)
            if show is not None:
                show_desc = re.sub(r'https?://\S+|www\.\S+', '', show["description"])
                num = len(show["episodes"]["items"])-1
                for i in range(min(num, 10)):
                    episode_desc = re.sub(r'https?://\S+|www\.\S+', '', show["episodes"]["items"][i]["description"])
                    episode_name = show["episodes"]["items"][i]["name"]
                    input = f"The show description is: {show_desc} The episode description is: {episode_desc}"
                    dict["Podcast Name"].append(name)
                    dict["Podcast ID"].append(id)
                    dict["Episode Name"].append(episode_name)
                    dict["Description"].append(input)
        l=1
    return dict

def search_songs(csv_file):
    keys_ = ['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    song_dict = {key: [] for key in keys_}

    df = pd.read_csv(csv_file)
    songs, artists, genres = df["name"].tolist(), df["artist"].tolist(), df["genre"].tolist()

    not_found = 0
    for i in range(len(songs)):
        if i % 20 == 0: print(i)
        time.sleep(1)
        song, artist = songs[i], artists[i]
        if i == 0: print(song)
        r = get_song(song, artist)
        try:
            track_id = r['tracks']['items'][0]['id']
            data = get_audio_analysis(track_id, headers)
            song_dict["id"].append(track_id)
            for key in keys_[1:]:
                song_dict[key].append(data[key])
        except:
            print(f"Could not find {song} by {artist}")
            for key in keys_:
                song_dict[key].append("")
            not_found += 1

    data = {'name':songs, 'artist':artists, 'genre':genres}
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(song_dict)
    df = pd.concat([df1, df2], axis=1)
    output_file = csv_file.split(".csv")[0] + "__audio_features.csv"
    df.to_csv(output_file, index=False)  


"""
podcasts = read_csv_podcasts("/top_podcasts.csv")
df = pd.DataFrame.from_dict(podcasts)
df.to_csv('podcasts.csv', index=False)  

s = get_random_songs("/top_artists/Top_Artists_4.csv", headers)
genre, song = s.random_songs()
df = random_songs_2(20); print(df)
"""