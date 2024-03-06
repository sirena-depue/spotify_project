import requests
import base64
import webbrowser
from urllib.parse import urlencode

def get_code(url_str):
    code = url_str.split("code=")[1].split("&state")[0]
    return code

def authorize_spotify():
    from spotify_config import client_id, client_secret
    redirect = 'insert redirect url here'
    params = {
                'response_type': 'code',  
                'client_id': client_id,
                'redirect_uri': redirect,
                'scope': 'user-read-private user-read-email',
                'state': 16}

    response = requests.get('https://accounts.spotify.com/authorize', params=params, allow_redirects=True)
    response.raise_for_status()
    full_authorization_url = 'https://accounts.spotify.com/authorize?' + urlencode(params)
    webbrowser.open(full_authorization_url)

    #### AUTHORIZE ####
    url = input("Paste the url:")

    code = get_code(url)
    encoded_credentials = base64.b64encode(client_id.encode() + b':' + client_secret.encode()).decode("utf-8")
    token_headers = {
        "Authorization": "Basic " + encoded_credentials,
        "Content-Type": "application/x-www-form-urlencoded"}

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect}

    refresh_token_data = {
        "grant_type": "refresh_token",
        "refresh_token": code,
        "client_id": client_id}

    r = requests.post("https://accounts.spotify.com/api/token", data=token_data, headers=token_headers)
    token = r.json()
    access_token = token.get('access_token')
    refresh_token = token.get('refresh_token')

    file_path = "token.txt"
    with open(file_path, 'w') as file:
        file.write(f"access: {access_token}")
        file.write(f"\nrefresh: {refresh_token}")
    print("updated token.txt")

    """
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'limit': 50, 'offset': 0}
    response = requests.get("https://api.spotify.com/v1/me/tracks", headers=headers, params=params)
    print(response.json())

    user_headers = {"Authorization": "Bearer " + access_token, "Content-Type": "application/json"}
    user_params = {"limit": 50}
    user_tracks_response = requests.get("https://api.spotify.com/v1/me/tracks", params=user_params, headers=user_headers)
    print(user_tracks_response.json())
    """
authorize_spotify()
