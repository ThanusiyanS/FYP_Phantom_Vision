from googleapiclient.discovery import build
import os
from pytube import YouTube
from moviepy import VideoFileClip
import subprocess
import csv

YOUTUBE_API_KEY = "AIzaSyA0tzzM9_60mvsABUUUc31HC-wuXsr8kHc"


def search_youtube(query, max_results=5):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=max_results,
        type="video"
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

def download_youtube_videos(query, max_results=5, download_dir="data/retrieved-videos", audio_dir="data/extracted-audios", csv_path="data/video_data.csv"):
    # Clear previous data
    clear_folder(download_dir)
    clear_folder(audio_dir)
    # Reset CSV file
    if os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ["video_id", "video_url", "video_path", "probability", "is_accident"]
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
    for idx, video in enumerate(videos, 1):
        video_path = download_video_yt_dlp(video["url"], download_dir)
        video_id = f"vid_{idx:02d}"
        probability = None
        is_accident = None
        if video_path:
            # Extract audio as before
            audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
            audio_path = os.path.join(audio_dir, audio_filename)
            try:
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(audio_path)
                clip.close()
                print(f"Extracted audio for {os.path.basename(video_path)}")
            except Exception as e:
                print(f"Failed to extract audio for {os.path.basename(video_path)}: {e}")
                audio_path = None
            results.append({"video_path": video_path, "audio_path": audio_path})
            csv_rows.append({
                "video_id": video_id,
                "video_url": video["url"],
                "video_path": video_path,
                "probability": probability,
                "is_accident": is_accident
            })
        else:
            print(f"Failed to download {video['title']}")
    # Write to CSV (append if exists, else create with header)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ["video_id", "video_url", "video_path", "probability", "is_accident"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    return results
