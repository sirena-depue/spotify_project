from lyricsgenius import Genius
from openai import OpenAI
import pandas as pd
import time
import re

"""
User needs to create genius_config.py and openai_config files with keys 
GENIUS API: https://docs.genius.com/#/getting-started-h1
OPENAI API: https://openai.com/blog/openai-api
"""
from genius_config import client_secret
from openai_config import api_key
client = OpenAI(api_key=api_key)
genius = Genius(client_secret)
genius.remove_section_headers = True

def song_descriptions_audio_features(songs_file):
    """ Function to create description of songs from Spotify audio features
        User can change the cutoff values for what is considered low/high in the "c" and "l" variables below, and corresponding descriptions
        Creates new column for "description" and saves to new csv file
        Inputs:
            - songs_file: name of csv file for song descriptions. Requires spotify audio features 
    """
    c = [0.3, 0.7]
    l = [-40, 10]
    """     AUDIO FEATURE        LESS THAN                        BETWEEN                             GREATER THAN                         """
    vars = {"loudness":         {"<":[l[0], "loud"],              "<>" : None,                        ">":[l[1], "soft/quiet"]},
            "valence":          {"<":[c[0], "negative"],          "<>" : None,                        ">":[c[1], "upbeat/positive"]},
            "energy":           {"<":[c[0], "low energy"],        "<>" : "moderate energy",           ">":[c[1], "high energy"]},
            "danceability":     {"<":[c[0], "not danceable"],     "<>" : "moderately danceable",      ">":[c[1], "very danceable"]},
            "instrumentalness": {"<":[c[0], "low instrumental"],  "<>" : "moderate instrumentalness", ">":[c[1], "high instrumentalness"]},
            "acousticness":     {"<":[c[0], "low acousticness"],  "<>" : "moderate acousticness",     ">":[c[1], "high acousticness"]}}

    df = pd.read_csv(songs_file)
    descriptions = []
    for _, row in df.iterrows():
        song_name, genre = row["name"], row["genre"].lower()
        description = f"The song {song_name} is in the {genre} genre. It is a "
        for i, var in enumerate(vars.keys()):
            variable = row[var]
            less, between, greater = vars[var]["<"], vars[var]["<>"], vars[var][">"] 
            cutoff_less, desc_less = less[0], less[1]
            cutoff_greater, desc_greater = greater[0], greater[1]
            if variable < cutoff_less:
                if i == 5: description += f"{desc_less} song."
                else: description += f"{desc_less}, "
            if between is not None:
                if cutoff_less <= variable < cutoff_greater:
                    if i == 5: description += f"{between} song."
                    else: description += f"{between}, "
            if variable > cutoff_greater:
                if i == 5: description += f"{desc_greater} song."
                else: description += f"{desc_greater}, "
        descriptions.append(description)
    df["description"] = descriptions
    output_file = songs_file.split(".csv")[0] + "_description.csv"
    df.to_csv(output_file, index=False)      

def get_lyrics(songs_file):
    """ Function to search for song lyrics using Genius API
        Creates new column for lyrics and saves to new csv file
        Inputs:
            - songs_file: name of csv file for song descriptions
    """
    def request_genius(song, artist):
        """ Helper function to call Genius API (prone to reaching limits)"""
        for _ in range(5):
            try:
                return genius.search_song(song, artist)
            except:
                print(f"Sleeping for 30 seconds ...")
                time.sleep(30)  
        return None
    def process_lyrics(song):
        """ Helper function to do some basic text processing including removing section headers """
        lyrics = song.lyrics.lower()
        lyrics = lyrics.split('\n')
        fin = ""
        for i,line in enumerate(lyrics):
            if "[" not in line and "]" not in line:
                if i == len(lyrics)-1: 
                    pattern = r'\d'
                    fin += re.split(pattern, line, maxsplit=1)[0]
                else:
                    if " contributors" not in line.lower():
                        fin += line + " "
        return fin
    df = pd.read_csv(songs_file)
    lyrics = []
    songs, artists = df["name"], df["artist"]
    for i in range(len(songs)): 
        if i % 20 == 0: print(i)
        song = request_genius(songs[i], artists[i])
        try:
            l = process_lyrics(song)
        except: 
            l = ""
        lyrics.append(l)
    df["lyrics"] = lyrics
    output_file = songs_file.split(".csv")[0] + "_lyrics.csv"
    df.to_csv(output_file, index=False)  

def summarize_lyrics(songs_file):
    """ Function to prompt ChatGPT to summarize song lyrics. 
        Creates new column for summaries and saves to new csv file
        Inputs:
            - songs_file: name of csv file for song descriptions
    """
    def request_chatgpt(prompt):
        """ Helper function to request output from chatgpt. Processes the output and returns the response.
            Input:
                - prompt: processed songs lyrics
        """
        completion = client.chat.completions.create(
            model= 'gpt-3.5-turbo-1106', # other options include: 3.5-turbo-0125 (cheap!), gpt-4 (expensive!!!)
            messages = [{"role": "user", "content": prompt}]
        )
        ans = str(completion.choices[0]).split("content=")[1][1:]
        ans = ans.split(", role")[0][:-1]
        return ans
    
    df = pd.read_csv(songs_file)
    lyrics = df["lyrics"].tolist()
    descs = []
    for i, lyric in enumerate(lyrics):
        prompt = f"How would you summarize the lyrics to this song: {lyric}"
        try:
            summary = request_chatgpt(prompt)
        except:
            print(f"No Response at i={i}")
            summary = ""
        if i % 20 == 0: print(i)
        descs.append(summary)
    df["new description"] = descs
    output_file = songs_file.split(".csv")[0] + "_new_description.csv"
    df.to_csv(output_file, index=False)  


songs_file = "songs.csv"

# BELOW: call to create description of songs from Spotify audio features
song_descriptions_audio_features(songs_file)  

# BELOW: call to search for song lyrics using Genius API
get_lyrics(songs_file) 

# BELOW: call to prompt ChatGPT to summarize song lyrics
summarize_lyrics(songs_file)

