import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from dotenv import load_dotenv
import io
import numpy as np
from matplotlib.ticker import MaxNLocator
import math
import argparse
import time

def create_visualizations(history, graphs_dir):
    """Create all the standard visualizations and save them to the graphs directory."""
    # --- Visualization 1: Top Artists ---
    top_artists = history.groupby('artist')['minutes_played'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_artists.values, y=top_artists.index, palette="viridis")
    plt.title('Top 10 Artists by Listening Time (minutes)')
    plt.xlabel('Minutes Played')
    plt.ylabel('Artist')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_artists.png'))
    plt.close()

    # --- Visualization 2: Top Tracks ---
    top_tracks = history.groupby(['track_name', 'artist', 'track_id'])['minutes_played'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_tracks.values, y=[f"{t[0]}\n({t[1]})" for t in top_tracks.index], palette="magma")
    plt.title('Top 10 Tracks by Listening Time (minutes)')
    plt.xlabel('Minutes Played')
    plt.ylabel('Track (Artist)')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_tracks.png'))
    plt.close()

    # --- Visualization 3: Listening Over Time ---
    daily = history.groupby('date')['minutes_played'].sum()
    plt.figure(figsize=(12,6))
    daily.plot()
    plt.title('Listening Time per Day')
    plt.xlabel('Date')
    plt.ylabel('Minutes Played')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_over_time.png'))
    plt.close()

    # --- Visualization 4: Listening by Hour ---
    hourly = history.groupby('hour')['minutes_played'].sum()
    plt.figure(figsize=(10,6))
    sns.barplot(x=hourly.index, y=hourly.values, palette="coolwarm")
    plt.title('Listening by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Minutes Played')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_by_hour.png'))
    plt.close()

    # --- Visualization 5: Artist Diversity ---
    artist_counts = history['artist'].value_counts().head(10)
    plt.figure(figsize=(8,8))
    plt.pie(artist_counts.values, labels=artist_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Top 10 Artists by Number of Plays')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'artist_diversity.png'))
    plt.close()

    # --- Combine all PNGs into a single PDF ---
    image_files = [
        'top_artists.png',
        'top_tracks.png',
        'listening_over_time.png',
        'listening_by_hour.png',
        'artist_diversity.png'
    ]
    images = [Image.open(os.path.join(graphs_dir, img)).convert('RGB') for img in image_files if os.path.exists(os.path.join(graphs_dir, img))]
    if images:
        images[0].save(os.path.join(graphs_dir, 'spotify_analysis.pdf'), save_all=True, append_images=images[1:])

    # --- Additional Visualizations ---
    # Listening Heatmap
    history['weekday'] = history['timestamp'].dt.day_name()
    heatmap_data = history.groupby(['weekday', 'hour'])['minutes_played'].sum().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.figure(figsize=(12,6))
    sns.heatmap(heatmap_data, cmap='YlGnBu')
    plt.title('Listening Heatmap (Day of Week x Hour)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_heatmap.png'))
    plt.close()

    # Monthly/Weekly Trends
    history['month'] = history['timestamp'].dt.to_period('M')
    history['week'] = history['timestamp'].dt.to_period('W')
    monthly = history.groupby('month')['minutes_played'].sum()
    weekly = history.groupby('week')['minutes_played'].sum()
    
    plt.figure(figsize=(10,5))
    monthly.plot(kind='bar')
    plt.title('Total Listening Time per Month')
    plt.xlabel('Month')
    plt.ylabel('Minutes Played')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_by_month.png'))
    plt.close()
    
    plt.figure(figsize=(12,5))
    weekly.plot()
    plt.title('Total Listening Time per Week')
    plt.xlabel('Week')
    plt.ylabel('Minutes Played')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_by_week.png'))
    plt.close()

    # Artist/Track Diversity per Month
    artist_diversity = history.groupby('month')['artist'].nunique()
    track_diversity = history.groupby('month')['track_name'].nunique()
    plt.figure(figsize=(10,5))
    artist_diversity.plot(label='Unique Artists')
    track_diversity.plot(label='Unique Tracks')
    plt.title('Artist and Track Diversity per Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'diversity_per_month.png'))
    plt.close()

    # Time of Day Distribution
    history['time_of_day'] = pd.cut(history['hour'], 
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    time_dist = history.groupby('time_of_day')['minutes_played'].sum()
    
    plt.figure(figsize=(8,8))
    plt.pie(time_dist.values, labels=time_dist.index, autopct='%1.1f%%', 
            colors=sns.color_palette('pastel'))
    plt.title('Listening Time Distribution by Time of Day')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'time_of_day_distribution.png'))
    plt.close()

    # Artist Discovery Over Time
    history['month'] = history['timestamp'].dt.to_period('M')
    new_artists_per_month = history.groupby('month')['artist'].nunique()
    cumulative_artists = history.groupby('month')['artist'].nunique().cumsum()
    
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()
    
    ax1.bar(new_artists_per_month.index.astype(str), new_artists_per_month.values, 
            alpha=0.3, label='New Artists')
    ax2.plot(cumulative_artists.index.astype(str), cumulative_artists.values, 
             'r-', label='Cumulative Artists')
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('New Artists per Month')
    ax2.set_ylabel('Cumulative Unique Artists')
    
    plt.title('Artist Discovery Over Time')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'artist_discovery.png'))
    plt.close()

    # Top Artists' Popularity Over Time
    top_artists_monthly = history[history['artist'].isin(top_artists.index[:5])].groupby(['month', 'artist'])['minutes_played'].sum().unstack()
    
    plt.figure(figsize=(12,6))
    top_artists_monthly.plot()
    plt.title('Top 5 Artists\' Listening Time Over Months')
    plt.xlabel('Month')
    plt.ylabel('Minutes Played')
    plt.legend(title='Artist')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_artists_over_time.png'))
    plt.close()

    # Top Tracks' Popularity Over Time
    top_tracks_monthly = history.groupby(['track_name', 'artist', 'month'])['minutes_played'].sum().reset_index()
    top_tracks_monthly['track_artist'] = top_tracks_monthly['track_name'] + ' (' + top_tracks_monthly['artist'] + ')'
    top_tracks_total = top_tracks_monthly.groupby(['track_name', 'artist'])['minutes_played'].sum().sort_values(ascending=False)
    top_10_tracks = top_tracks_total.head(10).index
    
    plt.figure(figsize=(15,8))
    for track_name, artist in top_10_tracks:
        track_data = top_tracks_monthly[
            (top_tracks_monthly['track_name'] == track_name) & 
            (top_tracks_monthly['artist'] == artist)
        ]
        plt.plot(track_data['month'].astype(str), track_data['minutes_played'], 
                marker='o', label=f"{track_name}\n({artist})")
    
    plt.title('Top 10 Tracks\' Listening Time Over Months')
    plt.xlabel('Month')
    plt.ylabel('Minutes Played')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_tracks_over_time.png'))
    plt.close()

    # Track Discovery Heatmap
    first_play = history.groupby(['track_name', 'artist'])['timestamp'].min().reset_index()
    first_play['month'] = first_play['timestamp'].dt.to_period('M')
    monthly_discoveries = first_play.groupby('month').size()
    
    plt.figure(figsize=(12,6))
    monthly_discoveries.plot(kind='bar')
    plt.title('Number of New Tracks Discovered Each Month')
    plt.xlabel('Month')
    plt.ylabel('Number of New Tracks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'track_discovery.png'))
    plt.close()

    # Track Replay Patterns
    track_replays = history.groupby(['track_name', 'artist', 'month']).size().reset_index(name='plays')
    track_replays['track_artist'] = track_replays['track_name'] + ' (' + track_replays['artist'] + ')'
    
    top_tracks_by_plays = track_replays.groupby('track_artist')['plays'].sum().sort_values(ascending=False).head(5)
    
    plt.figure(figsize=(12,6))
    for track in top_tracks_by_plays.index:
        track_data = track_replays[track_replays['track_artist'] == track]
        plt.plot(track_data['month'].astype(str), track_data['plays'], 
                marker='o', label=track)
    
    plt.title('Monthly Play Count for Top 5 Most Played Tracks')
    plt.xlabel('Month')
    plt.ylabel('Number of Plays')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'track_replay_patterns.png'))
    plt.close()

    # --- New Visualizations ---
    
    # Listening Streaks Analysis
    print("Creating listening streaks analysis...")
    daily_listening = history.groupby('date')['minutes_played'].sum()
    daily_listening = daily_listening[daily_listening > 0]  # Only consider days with listening
    
    # Calculate streaks
    dates = pd.Series(daily_listening.index)
    streak_lengths = []
    current_streak = 1
    
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            current_streak += 1
        else:
            streak_lengths.append(current_streak)
            current_streak = 1
    streak_lengths.append(current_streak)
    
    plt.figure(figsize=(10,6))
    sns.histplot(streak_lengths, bins=range(1, max(streak_lengths) + 2))
    plt.title('Distribution of Consecutive Days of Listening')
    plt.xlabel('Streak Length (days)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_streaks.png'))
    plt.close()

    # Repeat Listening Patterns
    print("Creating repeat listening patterns...")
    track_repeats = history.groupby(['date', 'track_name', 'artist']).size().reset_index(name='plays')
    repeat_distribution = track_repeats['plays'].value_counts().sort_index()
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=repeat_distribution.index, y=repeat_distribution.values)
    plt.title('Distribution of Track Repeats Within a Day')
    plt.xlabel('Number of Times Played in a Day')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'repeat_patterns.png'))
    plt.close()

    # First/Last Time Listened for Top Artists
    print("Creating first/last time analysis for top artists...")
    top_artists = history.groupby('artist')['minutes_played'].sum().sort_values(ascending=False).head(5).index
    artist_first_last = history[history['artist'].isin(top_artists)].groupby('artist').agg({
        'timestamp': ['min', 'max']
    })
    
    plt.figure(figsize=(12,6))
    for artist in top_artists:
        first = artist_first_last.loc[artist, ('timestamp', 'min')]
        last = artist_first_last.loc[artist, ('timestamp', 'max')]
        plt.plot([first, last], [artist, artist], 'o-', label=artist)
    
    plt.title('First and Last Time Listened for Top 5 Artists')
    plt.xlabel('Date')
    plt.ylabel('Artist')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'artist_first_last.png'))
    plt.close()

    # Listening Gaps Analysis
    print("Creating listening gaps analysis...")
    dates = pd.Series(daily_listening.index)
    gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    
    plt.figure(figsize=(10,6))
    sns.histplot(gaps, bins=range(0, max(gaps) + 2))
    plt.title('Distribution of Days Between Listening Sessions')
    plt.xlabel('Days Between Sessions')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_gaps.png'))
    plt.close()

    # Session Length Distribution
    print("Creating session length analysis...")
    # Define a session as plays within 30 minutes of each other
    history = history.sort_values('timestamp')
    history['time_diff'] = history['timestamp'].diff()
    history['new_session'] = history['time_diff'] > pd.Timedelta(minutes=30)
    history['session_id'] = history['new_session'].cumsum()
    
    session_lengths = history.groupby('session_id')['minutes_played'].sum()
    
    plt.figure(figsize=(10,6))
    sns.histplot(session_lengths, bins=50)
    plt.title('Distribution of Listening Session Lengths')
    plt.xlabel('Session Length (minutes)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'session_lengths.png'))
    plt.close()

    # Update the PDF with all visualizations
    all_image_files = [
        # Original visualizations
        'top_artists.png',
        'top_tracks.png',
        'listening_over_time.png',
        'listening_by_hour.png',
        'artist_diversity.png',
        'listening_heatmap.png',
        'listening_by_month.png',
        'listening_by_week.png',
        'diversity_per_month.png',
        'time_of_day_distribution.png',
        'artist_discovery.png',
        'top_artists_over_time.png',
        'top_tracks_over_time.png',
        'track_discovery.png',
        'track_replay_patterns.png'
    ]
    
    all_images = [Image.open(os.path.join(graphs_dir, img)).convert('RGB') 
                 for img in all_image_files if os.path.exists(os.path.join(graphs_dir, img))]
    if all_images:
        all_images[0].save(os.path.join(graphs_dir, 'spotify_analysis_detailed.pdf'), 
                          save_all=True, append_images=all_images[1:])

def fetch_album_covers(history, graphs_dir):
    """Fetch and create a grid of album covers for all unique tracks."""
    try:
        load_dotenv()
        CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
        CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

        if not CLIENT_ID or not CLIENT_SECRET:
            print("Error: Spotify API credentials not found in .env file")
            return

        def get_access_token(client_id, client_secret):
            token_url = "https://accounts.spotify.com/api/token"
            payload = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret
            }
            response = requests.post(token_url, data=payload)
            print(f"\nToken request status: {response.status_code}")
            print(f"Token response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"Error getting access token: {response.status_code}")
                print("Response:", response.text)
                return None
                
            token_data = response.json()
            print(f"Token expires in: {token_data.get('expires_in', 'unknown')} seconds")
            return token_data["access_token"]

        access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
        if not access_token:
            print("Failed to get access token")
            return

        headers = {"Authorization": f"Bearer {access_token}"}
        print("\nUsing headers:", headers)

        class RateLimiter:
            def __init__(self, max_requests=10, window_seconds=60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = []
            
            def wait_if_needed(self):
                now = time.time()
                # Remove requests older than the window
                self.requests = [req_time for req_time in self.requests 
                                if now - req_time < self.window_seconds]
                
                if len(self.requests) >= self.max_requests:
                    # Calculate how long to wait
                    oldest_request = self.requests[0]
                    wait_time = self.window_seconds - (now - oldest_request)
                    # Add a small buffer to be safe
                    wait_time += 2
                    print(f"\nRate limit approaching. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Update now after waiting
                    now = time.time()
                
                # Add current request
                self.requests.append(now)
                # Keep requests sorted
                self.requests.sort()

        def make_request(url, max_retries=5):
            """Make a request with exponential backoff retry logic and rate limiting."""
            for attempt in range(max_retries):
                try:
                    rate_limiter.wait_if_needed()
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 429:  # Rate limit hit
                        retry_after = int(response.headers.get('Retry-After', 30))
                        retry_after = min(retry_after, 60)  # Cap at 60 seconds
                        print(f"\nRate limit hit. Response body: {response.text}")
                        print(f"Waiting {retry_after} seconds before retry...")
                        time.sleep(retry_after)
                        continue
                    elif response.status_code != 200:
                        print(f"\nError response (attempt {attempt + 1}/{max_retries}):")
                        print(f"Status code: {response.status_code}")
                        print(f"Response: {response.text}")
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 2  # Exponential backoff
                            print(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                        return None
                        
                    return response
                except Exception as e:
                    print(f"\nRequest failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        raise e
            return None

        # Initialize rate limiter with more conservative limits
        rate_limiter = RateLimiter(max_requests=8, window_seconds=60)

        # Get all unique tracks and their album IDs
        print("\nFetching album information for unique tracks...")
        unique_tracks = history[['track_name', 'artist', 'track_id']].drop_duplicates()
        total_tracks = len(unique_tracks)
        print(f"Found {total_tracks} unique tracks")
        
        album_covers = {}  # Dictionary to store album_id -> image mapping
        track_to_album = {}  # Dictionary to store track_id -> album_id mapping

        # Process tracks in smaller batches of 20
        batch_size = 20
        for i in range(0, len(unique_tracks), batch_size):
            batch = unique_tracks.iloc[i:i+batch_size]
            track_ids = batch['track_id'].dropna().tolist()
            
            if not track_ids:
                continue
                
            progress = (i + len(batch)) / total_tracks * 100
            print(f"\nProcessing tracks: {progress:.1f}% complete")
            print(f"Batch {i//batch_size + 1}/{(total_tracks + batch_size - 1)//batch_size}")
            
            # Create comma-separated string of track IDs
            track_ids_str = ','.join(track_ids)
            url = f"https://api.spotify.com/v1/tracks?ids={track_ids_str}"
            
            try:
                r = make_request(url)
                if r and r.status_code == 200:
                    data = r.json()
                    for track in data.get('tracks', []):
                        if track and 'album' in track and 'id' in track['album']:
                            track_id = track['id']
                            album_id = track['album']['id']
                            track_to_album[track_id] = album_id
                            print(f"Successfully got album ID for track {track_id}")
                        else:
                            print(f"No album data for track {track.get('id', 'unknown')}")
                else:
                    print(f"Failed to fetch batch of tracks")
                    if r:
                        print(f"Status code: {r.status_code}")
                        print(f"Response: {r.text}")
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add a longer wait after an error
                time.sleep(10)

        # Get unique album IDs
        unique_albums = list(set(track_to_album.values()))
        print(f"\nFound {len(unique_albums)} unique albums out of {total_tracks} tracks")
        
        if not unique_albums:
            print("No album IDs found. Check if track IDs are valid and API credentials are correct.")
            return

        # Second pass: Fetch album covers in smaller batches of 10
        batch_size = 10
        for i in range(0, len(unique_albums), batch_size):
            batch = unique_albums[i:i+batch_size]
            progress = (i + len(batch)) / len(unique_albums) * 100
            print(f"\nFetching album covers: {progress:.1f}% complete")
            print(f"Batch {i//batch_size + 1}/{(len(unique_albums) + batch_size - 1)//batch_size}")
            
            # Create comma-separated string of album IDs
            album_ids = ','.join(batch)
            url = f"https://api.spotify.com/v1/albums?ids={album_ids}"
            
            try:
                r = make_request(url)
                if r and r.status_code == 200:
                    data = r.json()
                    for album in data['albums']:
                        if album and album['images']:
                            try:
                                img_url = album['images'][0]['url']
                                img_data = requests.get(img_url).content
                                img = Image.open(io.BytesIO(img_data)).resize((100, 100))
                                album_covers[album['id']] = img
                            except Exception as e:
                                print(f"\nError loading image for album {album['id']}: {e}")
                                album_covers[album['id']] = Image.new('RGB', (100, 100), color='gray')
                        else:
                            print(f"\nNo images found for album {album['id'] if album else 'unknown'}")
                            album_covers[album['id']] = Image.new('RGB', (100, 100), color='gray')
                else:
                    print(f"\nFailed to fetch batch of albums")
                    if r:
                        print(f"Status code: {r.status_code}")
                        print(f"Response: {r.text}")
                    # Add gray placeholders for failed batch
                    for album_id in batch:
                        album_covers[album_id] = Image.new('RGB', (100, 100), color='gray')
            except Exception as e:
                print(f"\nError processing batch: {str(e)}")
                # Add gray placeholders for failed batch
                for album_id in batch:
                    album_covers[album_id] = Image.new('RGB', (100, 100), color='gray')
                # Add a longer wait after an error
                time.sleep(10)
        
        print("\nCreating album cover grid...")
        
        # Create the grid using album covers
        if album_covers:
            # Calculate grid dimensions
            n_albums = len(album_covers)
            cols = int(math.ceil(math.sqrt(n_albums * 1.5)))  # Use 1.5 ratio for better visual layout
            rows = int(math.ceil(n_albums / cols))
            
            # Create the grid image
            grid_img = Image.new('RGB', (100*cols, 100*rows), color='white')
            for idx, (album_id, img) in enumerate(album_covers.items()):
                x = (idx % cols) * 100
                y = (idx // cols) * 100
                grid_img.paste(img, (x, y))
            
            # Save the grid
            grid_img.save(os.path.join(graphs_dir, 'all_album_covers.png'))
            print(f"Saved album cover grid with {n_albums} unique albums to {os.path.join(graphs_dir, 'all_album_covers.png')}")
        else:
            print("No album covers were successfully fetched")

    except Exception as e:
        print(f"Error in album cover fetching: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze Spotify listening history and optionally create album cover visualization.')
    parser.add_argument('--fetch-covers', action='store_true', help='Fetch and create album cover visualization')
    args = parser.parse_args()

    # Set style for seaborn
    sns.set(style="whitegrid")

    # Create graphs directory if it doesn't exist
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    # Directory containing the CSV files (assume current directory)
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all CSV files matching the pattern
    csv_files = sorted(glob.glob(os.path.join(data_dir, 'spotify_history_*.csv')))

    # Read and concatenate all CSV files
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, header=None, names=["timestamp", "track_id", "track_name", "artist", "ms_played"])
        df_list.append(df)

    history = pd.concat(df_list, ignore_index=True)

    # Convert timestamp to datetime
    history['timestamp'] = pd.to_datetime(history['timestamp'], format='ISO8601')

    # Add additional time columns
    history['date'] = history['timestamp'].dt.date
    history['hour'] = history['timestamp'].dt.hour

    # Convert ms_played to minutes
    history['minutes_played'] = history['ms_played'] / 60000

    # Create all standard visualizations
    print("Creating standard visualizations...")
    create_visualizations(history, graphs_dir)
    print("Standard visualizations complete!")

    # Optionally fetch album covers
    if args.fetch_covers:
        print("\nStarting album cover visualization...")
        fetch_album_covers(history, graphs_dir)
    else:
        print("\nSkipping album cover visualization. Use --fetch-covers to generate it.")

if __name__ == '__main__':
    main() 