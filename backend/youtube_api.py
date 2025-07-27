from googleapiclient.discovery import build
import os
from pytube import YouTube
from moviepy import VideoFileClip
import subprocess
import csv
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model

YOUTUBE_API_KEY = 


def search_youtube(query, max_results=5):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=max_results,
        type="video",
        videoDuration="short"  # Only get short videos
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"title": title, "url": url})

    return videos

def download_video_yt_dlp(url, save_dir="data/retrieved-videos"):
    os.makedirs(save_dir, exist_ok=True)
    command = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", os.path.join(save_dir, "%(title)s.%(ext)s"),
        url
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr}")
        return None
    else:
        # Find the downloaded file (yt-dlp prints the filename in stdout)
        for line in result.stdout.splitlines():
            if "[download] Destination:" in line:
                return line.split("Destination:")[1].strip()
        return None

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Remove ML and quality logic, only keep download and extraction
def download_youtube_videos(query, max_results=5, download_dir="data/retrieved-videos", audio_dir="data/extracted-audios", csv_path="data/initial_video_data.csv"):
    # Clear previous data
    clear_folder(download_dir)
    clear_folder(audio_dir)
    # Reset CSV file
    if os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ["video_id", "video_path", "audio_path", "only_video_path", "video_url"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    videos = search_youtube(query, max_results)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    results = []
    csv_rows = []
    urls = []
    video_names = []
    for idx, video in enumerate(videos, 1):
        video_id = f"item_{idx:02d}"
        audio_id = f"item_{idx:02d}"
        # Set new video filename
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(download_dir, video_filename)
        # Download video with yt-dlp and rename
        temp_video_path = download_video_yt_dlp(video["url"], download_dir)
        if not temp_video_path or not os.path.exists(temp_video_path):
            print(f"Skipping {video['title']} due to download failure.")
            continue
        if temp_video_path != video_path:
            os.rename(temp_video_path, video_path)
        if os.path.exists(video_path):
            # Extract audio and save as audio_id.mp3
            audio_filename = f"{audio_id}.mp3"
            audio_path = os.path.join(audio_dir, audio_filename)
            try:
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(audio_path)
                clip.close()
                print(f"Extracted audio for {video_filename}")
            except Exception as e:
                print(f"Failed to extract audio for {video_filename}: {e}")
                audio_path = None
            results.append({"video_id": video_id, "video_path": video_path, "audio_path": audio_path, "only_video_path": "", "video_url": video["url"]})
            csv_rows.append({
                "video_id": video_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "only_video_path": "",
                "video_url": video["url"]
            })
            # Add to urls and video_names for frontend response
            urls.append(video["url"])
            video_names.append(video["title"])
        else:
            print(f"Failed to download {video['title']}")
    # Write to CSV (append if exists, else create with header)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ["video_id", "video_path", "audio_path", "only_video_path", "video_url"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    return {"urls": urls, "video_names": video_names, "results": results}
