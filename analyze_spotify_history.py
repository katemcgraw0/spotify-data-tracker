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

def create_monthly_top_artists(history, graphs_dir):
    """Create a visualization showing top 5 artists for each month."""
    # Get top 5 artists for each month
    monthly_top_artists = history.groupby(['month', 'artist'])['minutes_played'].sum().reset_index()
    monthly_top_artists = monthly_top_artists.sort_values(['month', 'minutes_played'], ascending=[True, False])
    monthly_top_artists = monthly_top_artists.groupby('month').head(5)
    
    # Create a pivot table for easier plotting
    pivot_data = monthly_top_artists.pivot(index='artist', columns='month', values='minutes_played')
    
    # Sort months chronologically
    pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot each month's data
    x = np.arange(len(pivot_data.index))
    width = 0.15  # Width of bars
    months = pivot_data.columns
    
    for i, month in enumerate(months):
        values = pivot_data[month].fillna(0)
        plt.bar(x + (i * width), values, width, label=month.strftime('%B %Y'))
    
    # Customize the plot
    plt.xlabel('Artist')
    plt.ylabel('Minutes Played')
    plt.title('Top 5 Artists by Month')
    plt.xticks(x + width * 2, pivot_data.index, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graphs_dir, 'monthly_top_artists.png'))
    plt.close()

def create_monthly_artist_grid(history, graphs_dir):
    """Create a grid of artist images organized by month."""
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
            if response.status_code != 200:
                print(f"Error getting access token: {response.status_code}")
                return None
            return response.json()["access_token"]

        access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
        if not access_token:
            print("Failed to get access token")
            return

        headers = {"Authorization": f"Bearer {access_token}"}

        # Get top 5 artists for each month
        monthly_top_artists = history.groupby(['month', 'artist'])['minutes_played'].sum().reset_index()
        monthly_top_artists = monthly_top_artists.sort_values(['month', 'minutes_played'], ascending=[True, False])
        monthly_top_artists = monthly_top_artists.groupby('month').head(5)
        
        # Get unique artists
        unique_artists = monthly_top_artists['artist'].unique()
        artist_images = {}
        artist_to_id = {}  # Store artist name to ID mapping
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(graphs_dir, 'image_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Search for each artist and get their image
        for artist in unique_artists:
            try:
                # Check if we already have the artist ID
                if artist not in artist_to_id:
                    # Search for the artist
                    search_url = f"https://api.spotify.com/v1/search?q={requests.utils.quote(artist)}&type=artist&limit=1"
                    response = requests.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['artists']['items']:
                            artist_data = data['artists']['items'][0]
                            artist_to_id[artist] = artist_data['id']
                            print(f"Found ID for {artist}: {artist_data['id']}")
                        else:
                            print(f"No artist found for {artist}")
                            continue
                    else:
                        print(f"Error searching for {artist}: {response.status_code}")
                        continue
                
                artist_id = artist_to_id[artist]
                cache_path = os.path.join(cache_dir, f"{artist_id}.jpg")
                
                # Check if image exists in cache
                if os.path.exists(cache_path):
                    print(f"Using cached image for {artist}")
                    artist_images[artist] = cache_path
                else:
                    # Search for the artist to get their image
                    search_url = f"https://api.spotify.com/v1/search?q={requests.utils.quote(artist)}&type=artist&limit=1"
                    response = requests.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['artists']['items']:
                            artist_data = data['artists']['items'][0]
                            if artist_data['images']:
                                # Download and cache the image
                                img_url = artist_data['images'][0]['url']
                                img_response = requests.get(img_url)
                                if img_response.status_code == 200:
                                    with open(cache_path, 'wb') as f:
                                        f.write(img_response.content)
                                    print(f"Cached image for {artist}")
                                    artist_images[artist] = cache_path
                                else:
                                    print(f"Error downloading image for {artist}")
                            else:
                                print(f"No image found for {artist}")
                        else:
                            print(f"No artist found for {artist}")
                    else:
                        print(f"Error searching for {artist}: {response.status_code}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {artist}: {str(e)}")
                continue

        # Create the grid
        months = sorted(monthly_top_artists['month'].unique())
        # Make the figure smaller and adjust the spacing
        fig, axes = plt.subplots(len(months), 5, figsize=(10, 2*len(months)))
        fig.subplots_adjust(wspace=0, hspace=0)  # Remove spacing between subplots
        
        for i, month in enumerate(months):
            month_artists = monthly_top_artists[monthly_top_artists['month'] == month]
            
            for j, (_, row) in enumerate(month_artists.iterrows()):
                ax = axes[i, j]
                artist = row['artist']
                
                if artist in artist_images:
                    try:
                        # Load image from cache
                        img = Image.open(artist_images[artist])
                        ax.imshow(img)
                        
                        # Add ranking number in top-left corner
                        ax.text(0.05, 0.95, f"#{j+1}", 
                               ha='left', va='top', 
                               transform=ax.transAxes,
                               color='white',
                               fontsize=10,
                               fontweight='bold',
                               bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
                        
                        # Add artist name at bottom
                        ax.text(0.5, 0.05, artist, 
                               ha='center', va='bottom', 
                               transform=ax.transAxes,
                               color='white',
                               fontsize=8,
                               bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
                    except Exception as e:
                        print(f"Error loading image for {artist}: {e}")
                        ax.text(0.5, 0.5, artist, ha='center', va='center', wrap=True)
                else:
                    ax.text(0.5, 0.5, artist, ha='center', va='center', wrap=True)
                
                ax.axis('off')
                if j == 0:  # Add month label only for first column
                    # Create a background for the month label
                    ax.text(-0.1, 0.5, month.strftime('%B %Y'), 
                           ha='right', va='center', 
                           transform=ax.transAxes,
                           color='white',
                           fontsize=10,
                           fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, 'monthly_artist_grid.png'), 
                   bbox_inches='tight', 
                   dpi=300,
                   pad_inches=0)
        plt.close()
        
    except Exception as e:
        print(f"Error in monthly artist grid creation: {str(e)}")
        import traceback
        traceback.print_exc()

def create_visualizations(history, graphs_dir):
    """Create all the standard visualizations and save them to the graphs directory."""
    # Set consistent style for all plots
    plt.style.use('seaborn-v0_8')  # Use a valid style name
    sns.set_theme(style="whitegrid")  # Set seaborn theme
    
    # Common figure sizes
    BAR_FIGSIZE = (12, 6)
    PIE_FIGSIZE = (10, 10)
    LINE_FIGSIZE = (15, 8)
    
    # Common formatting
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Add month column for time-based analysis
    history['month'] = history['timestamp'].dt.to_period('M')

    # Create monthly top artists visualization
    create_monthly_top_artists(history, graphs_dir)

    # Create monthly artist grid visualization
    create_monthly_artist_grid(history, graphs_dir)

    # ARTISTS 
    # --- Visualization 1: Top Artists ---
    top_artists = history.groupby('artist')['minutes_played'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=BAR_FIGSIZE)
    sns.barplot(x=top_artists.values, y=top_artists.index, hue=top_artists.index, legend=False)
    plt.title('Top 10 Artists by Listening Time')
    plt.xlabel('Total Minutes Played')
    plt.ylabel('Artist')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_artists.png'))
    plt.close()

    # --- Visualization 2: Top 10 Artists by Number of Plays ---
    artist_counts = history['artist'].value_counts().head(10)
    plt.figure(figsize=BAR_FIGSIZE)
    sns.barplot(x=artist_counts.values, y=artist_counts.index, hue=artist_counts.index, legend=False)
    plt.title('Top 10 Artists by Number of Plays')
    plt.xlabel('Total Number of Plays')
    plt.ylabel('Artist')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'artist_diversity.png'))
    plt.close()

    # --- Visualization 3: Top Artists' Popularity Over Time ---
    top_artists_monthly = history[history['artist'].isin(top_artists.index[:5])].groupby(['month', 'artist'])['minutes_played'].sum().unstack()
    
    plt.figure(figsize=LINE_FIGSIZE)
    top_artists_monthly.plot(marker='o')
    plt.title('Top 5 Artists\' Listening Time Over Months')
    plt.xlabel('Month')
    plt.ylabel('Minutes Played')
    plt.legend(title='Artist', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_artists_over_time.png'))
    plt.close()

    # TRACKS
    # --- Visualization 4: Top Tracks by Number of Minutes Played ---
    top_tracks_minutes = history.groupby(['track_name', 'artist'])['minutes_played'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=BAR_FIGSIZE)
    track_labels = [f"{t[0]}\n({t[1]})" for t in top_tracks_minutes.index]
    sns.barplot(x=top_tracks_minutes.values, y=track_labels, hue=track_labels, legend=False)
    plt.title('Top 10 Tracks by Listening Time')
    plt.xlabel('Total Minutes Played')
    plt.ylabel('Track (Artist)')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_tracks_minutes.png'))
    plt.close()

    # --- Visualization 5: Top Tracks by Number of Plays ---
    top_tracks_plays = history.groupby(['track_name', 'artist']).size().sort_values(ascending=False).head(10)
    plt.figure(figsize=BAR_FIGSIZE)
    track_labels = [f"{t[0]}\n({t[1]})" for t in top_tracks_plays.index]
    sns.barplot(x=top_tracks_plays.values, y=track_labels, hue=track_labels, legend=False)
    plt.title('Top 10 Tracks by Number of Plays')
    plt.xlabel('Total Number of Plays')
    plt.ylabel('Track (Artist)')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_tracks_plays.png'))
    plt.close()

    # --- Visualization 6: Top Tracks Over Time ---
    top_tracks_monthly = history.groupby(['track_name', 'artist', 'month'])['minutes_played'].sum().reset_index()
    top_tracks_monthly['track_artist'] = top_tracks_monthly['track_name'] + ' (' + top_tracks_monthly['artist'] + ')'
    top_tracks_total = top_tracks_monthly.groupby(['track_name', 'artist'])['minutes_played'].sum().sort_values(ascending=False)
    top_10_tracks = top_tracks_total.head(10).index
    
    plt.figure(figsize=LINE_FIGSIZE)
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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'top_tracks_over_time.png'))
    plt.close()

    # GENERAL LISTENING
    # --- Visualization 7: Listening Over Time ---
    daily = history.groupby('date')['minutes_played'].sum()
    plt.figure(figsize=LINE_FIGSIZE)
    daily.plot(marker='o', alpha=0.7)
    plt.title('Daily Listening Time')
    plt.xlabel('Date')
    plt.ylabel('Minutes Played')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_over_time.png'))
    plt.close()

    # --- Visualization 8: Listening by Hour ---
    hourly = history.groupby('hour')['minutes_played'].sum()
    plt.figure(figsize=BAR_FIGSIZE)
    sns.barplot(x=hourly.index, y=hourly.values, hue=hourly.index, legend=False)
    plt.title('Listening Time by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Minutes Played')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_by_hour.png'))
    plt.close()

    # --- Visualization 9: Listening Heatmap ---
    history['weekday'] = history['timestamp'].dt.day_name()
    heatmap_data = history.groupby(['weekday', 'hour'])['minutes_played'].sum().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Minutes Played'})
    plt.title('Listening Time Heatmap (Day of Week × Hour)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_heatmap.png'))
    plt.close()

    # --- Visualization 10: Monthly/Weekly Trends ---
    history['week'] = history['timestamp'].dt.to_period('W')
    monthly = history.groupby('month')['minutes_played'].sum()
    weekly = history.groupby('week')['minutes_played'].sum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    monthly.plot(kind='bar', ax=ax1)
    ax1.set_title('Monthly Listening Time')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Minutes Played')
    ax1.grid(True, alpha=0.3)
    
    weekly.plot(ax=ax2)
    ax2.set_title('Weekly Listening Time')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Minutes Played')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'listening_by_month.png'))
    plt.close()

    # --- Visualization 11: Artist/Track Diversity per Month ---
    artist_diversity = history.groupby('month')['artist'].nunique()
    track_diversity = history.groupby('month')['track_name'].nunique()
    
    plt.figure(figsize=LINE_FIGSIZE)
    artist_diversity.plot(marker='o', label='Unique Artists', color='#1f77b4')
    track_diversity.plot(marker='o', label='Unique Tracks', color='#ff7f0e')
    plt.title('Artist and Track Diversity Over Time')
    plt.xlabel('Month')
    plt.ylabel('Number of Unique Items')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'diversity_per_month.png'))
    plt.close()

    # --- Visualization 12: New Artist Discovery Over Time ---
    new_artists_per_month = history.groupby('month')['artist'].nunique()
    
    plt.figure(figsize=LINE_FIGSIZE)
    new_artists_per_month.plot(kind='bar')
    plt.title('New Artists Discovered per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of New Artists')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'artist_discovery.png'))
    plt.close()

    # --- Visualization 13: Track Discovery Heatmap ---
    first_play = history.groupby(['track_name', 'artist'])['timestamp'].min().reset_index()
    first_play['month'] = first_play['timestamp'].dt.to_period('M')
    monthly_discoveries = first_play.groupby('month').size()
    
    plt.figure(figsize=LINE_FIGSIZE)
    monthly_discoveries.plot(kind='bar')
    plt.title('New Tracks Discovered per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of New Tracks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'track_discovery.png'))
    plt.close()

    # Update the PDF with all visualizations
    all_image_files = [
        'top_artists.png',
        'artist_diversity.png', 
        'top_artists_over_time.png',
        'top_tracks_minutes.png',
        'top_tracks_plays.png',
        'top_tracks_over_time.png',
        'diversity_per_month.png',
        'artist_discovery.png',
        'track_discovery.png',
        'listening_over_time.png', 
        'listening_by_hour.png',
        'listening_heatmap.png',
        'listening_by_month.png',
        'monthly_artist_grid.png',
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

def update_readme_with_visualizations(graphs_dir):
    """Update the README.md file with the generated visualizations."""
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
    
    # Read existing README content
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Find the end of the existing content
    if '## Visualizations' not in content:
        content += '\n\n## Visualizations\n\n'
    
    # Create visualization section with timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    viz_section = f'\n### Latest Analysis (Last Updated: {current_time})\n\n'
    
    # Add each visualization with a description
    viz_descriptions = {
        'monthly_artist_grid.png': 'Top 5 Artists by Month (with Artist Images)',
        'top_artists.png': 'Top 10 Artists by Listening Time',
        'artist_diversity.png': 'Top 10 Artists by Number of Plays',
        'top_artists_over_time.png': 'Top 5 Artists\' Listening Time Over Months',
        'top_tracks_minutes.png': 'Top 10 Tracks by Listening Time',
        'top_tracks_plays.png': 'Top 10 Tracks by Number of Plays',
        'top_tracks_over_time.png': 'Top 10 Tracks\' Listening Time Over Months',
        'diversity_per_month.png': 'Artist and Track Diversity Over Time',
        'artist_discovery.png': 'New Artists Discovered per Month',
        'track_discovery.png': 'New Tracks Discovered per Month',
        'listening_over_time.png': 'Daily Listening Time',
        'listening_by_hour.png': 'Listening Time by Hour of Day',
        'listening_heatmap.png': 'Listening Time Heatmap (Day of Week × Hour)',
        'listening_by_month.png': 'Monthly and Weekly Listening Time',
        'all_album_covers.png': 'Album Cover Grid'
    }
    
    for img_file, description in viz_descriptions.items():
        img_path = os.path.join(graphs_dir, img_file)
        if os.path.exists(img_path):
            # Get the relative path for the image
            rel_path = os.path.relpath(img_path, os.path.dirname(readme_path))
            viz_section += f'#### {description}\n\n'
            viz_section += f'![{description}]({rel_path})\n\n'
    
    # Update the README content
    if '### Latest Analysis' in content:
        # Replace existing visualization section
        parts = content.split('### Latest Analysis')
        content = parts[0] + viz_section
    else:
        # Append new visualization section
        content += viz_section
    
    # Write the updated content back to README
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print("Updated README.md with latest visualizations")

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
    
    # Update README with visualizations
    print("\nUpdating README with visualizations...")
    update_readme_with_visualizations(graphs_dir)
    print("README update complete!")

if __name__ == '__main__':
    main() 