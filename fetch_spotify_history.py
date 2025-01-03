import os
import csv
import requests
from dotenv import load_dotenv
import pandas as pd  # For timestamp conversion

# Load environment variables from .env file
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("SPOTIFY_REFRESH_TOKEN")
CSV_FILE = "spotify_history.csv"

# Fixed timestamp for the beginning of 2025
START_OF_2025_TIMESTAMP = int(pd.Timestamp("2025-01-01T00:00:00Z").timestamp() * 1000)

def get_access_token(client_id, client_secret, refresh_token):
    """
    Exchanges the refresh token for a new access token.
    """
    token_url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(token_url, data=payload)
    response.raise_for_status()
    response_json = response.json()
    return response_json["access_token"]

def get_recently_played_tracks(access_token, after=None, limit=50):
    """
    Calls the Spotify API to get recently played tracks.
    """
    url = "https://api.spotify.com/v1/me/player/recently-played"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "limit": limit
    }
    if after:
        params["after"] = after
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def get_last_played_timestamp(csv_file):
    """
    Reads the CSV file to find the most recent 'played_at' timestamp.
    """
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                # Extract the last row's played_at timestamp
                return rows[-1][0]
    except FileNotFoundError:
        # If the file doesn't exist, return None
        return None
    return None

def main():
    # 1. Get a fresh access token using the refresh token
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
    
    # 2. Determine the timestamp to fetch data after
    last_played_at = get_last_played_timestamp(CSV_FILE)
    if last_played_at:
        after_timestamp = int(pd.to_datetime(last_played_at).timestamp() * 1000)
    else:
        # Start from the beginning of 2025 if the file doesn't exist
        after_timestamp = START_OF_2025_TIMESTAMP

    total_new_tracks = 0

    # 3. Continuously fetch data until there are no more tracks
    while True:
        recently_played_data = get_recently_played_tracks(access_token, after=after_timestamp, limit=50)
        items = recently_played_data.get("items", [])
        
        # If no new tracks are returned, break the loop
        if not items:
            print("No more new tracks to fetch.")
            break

        # Append new tracks to the CSV
        with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for item in items:
                track = item["track"]
                played_at = item["played_at"]
                track_id = track["id"]
                track_name = track["name"].replace(",", " ")
                artist_name = track["artists"][0]["name"].replace(",", " ")
                duration_ms = track["duration_ms"]
                writer.writerow([played_at, track_id, track_name, artist_name, duration_ms])
                total_new_tracks += 1

        # Update the `after_timestamp` to the last fetched track's `played_at`
        after_timestamp = int(pd.to_datetime(items[-1]["played_at"]).timestamp() * 1000)
    
    print(f"Added {total_new_tracks} new tracks to {CSV_FILE}.")

if __name__ == "__main__":
    main()
