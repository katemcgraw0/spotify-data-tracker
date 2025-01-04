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

def get_last_played_timestamp():
    """
    Reads the last played timestamp across all monthly files to determine the latest timestamp.
    """
    latest_timestamp = None
    for month in range(1, 13):
        csv_file = f"spotify_history_{month:02d}.csv"
        try:
            data = pd.read_csv(csv_file, header=None)
            data.columns = ["played_at", "track_id", "track_name", "artist_name", "duration_ms"]
            data = data.sort_values("played_at")
            last_timestamp = data["played_at"].iloc[-1]
            if not latest_timestamp or pd.to_datetime(last_timestamp) > pd.to_datetime(latest_timestamp):
                latest_timestamp = last_timestamp
        except FileNotFoundError:
            continue
    return latest_timestamp

def write_track_to_monthly_file(track_data):
    """
    Writes a track to the appropriate monthly CSV file based on its played_at timestamp.
    """
    played_at = pd.to_datetime(track_data["played_at"])
    month = played_at.month
    csv_file = f"spotify_history_{month:02d}.csv"
    track_info = [
        track_data["played_at"],
        track_data["track"]["id"],
        track_data["track"]["name"].replace(",", " ").replace("\n", " ").replace("\r", " "),
        track_data["track"]["artists"][0]["name"].replace(",", " ").replace("\n", " ").replace("\r", " "),
        track_data["track"]["duration_ms"]
    ]

    # Append to the monthly file
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(track_info)


def main():
    # Get a fresh access token
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
    
    # Determine the timestamp to fetch data after
    last_played_at = get_last_played_timestamp()
    if last_played_at:
        after_timestamp = int(pd.to_datetime(last_played_at).timestamp() * 1000)
    else:
        after_timestamp = START_OF_2025_TIMESTAMP

    total_new_tracks = 0

    # Continuously fetch data
    recently_played_data = get_recently_played_tracks(access_token, after=after_timestamp, limit=50)
    items = recently_played_data.get("items", [])
    
    if not items:
        print("No new tracks found.")
        return

    # Reverse the list to process from oldest to newest
    items_reversed = list(reversed(items))

    for item in items_reversed:
        print("Writing track:", item["track"]["name"], "played at", item["played_at"])
        write_track_to_monthly_file(item)
        total_new_tracks += 1
    print(f"Added {total_new_tracks} new tracks across monthly files.")

if __name__ == "__main__":
    main()
