import os
import librosa
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import csv
import webrtcvad
import soundfile as sf
import tensorflow_hub as hub
import tensorflow as tf
import urllib.request
import csv as pycsv
from sklearn.cluster import KMeans
import pandas as pd
import os
os.environ["TFHUB_CACHE_DIR"] = "./tfhub_cache"  # Local folder to avoid system cache reuse


script_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(script_dir, "..","data","extracted-audios")
model_path = os.path.join(script_dir, "audio_deepfake_detection_model_v3.pth")

# === MODEL ===
class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# === IN-MEMORY CHUNK PREDICTION ===
def predict_audio_score_from_array(model, y, sr, device):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(mfcc_tensor)
        return output.item()

# === YAMNET SETUP ===
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
# YAMNet class map
YAMNET_CLASSES = None
if YAMNET_CLASSES is None:
    class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    response = urllib.request.urlopen(class_map_url)
    lines = [l.decode('utf-8') for l in response.readlines()]
    reader = pycsv.reader(lines)
    next(reader)  # skip header
    YAMNET_CLASSES = [row[2] for row in reader]

# === HUMAN VOICE DETECTION WITH YAMNET ===
def is_human_voice_chunk(y, sr):
    # Resample to 16kHz mono float32
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    # Run YAMNet
    scores, embeddings, spectrogram = yamnet_model(y)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_class = np.argmax(mean_scores)
    top_class_name = YAMNET_CLASSES[top_class].lower()
    # Check for 'speech' or 'human voice' in class name
    if 'speech' in top_class_name or 'human voice' in top_class_name:
        return True
    return False

def process_audio_for_deepfake_in_memory(audio_path, model, device, chunk_length=10):
    y, sr = librosa.load(audio_path, sr=None)
    chunk_samples = int(sr * chunk_length)
    non_voice_scores = []
    all_chunks_have_voice = True
    if len(y) < chunk_samples:
        if is_human_voice_chunk(y, sr):
            pass  # skip, all_chunks_have_voice remains True
        else:
            all_chunks_have_voice = False
            non_voice_scores.append(predict_audio_score_from_array(model, y, sr, device))
    else:
        for i in range(0, len(y), chunk_samples):
            chunk = y[i:i + chunk_samples]
            if is_human_voice_chunk(chunk, sr):
                continue  # skip, all_chunks_have_voice remains True
            else:
                all_chunks_have_voice = False
                non_voice_scores.append(predict_audio_score_from_array(model, chunk, sr, device))
    if all_chunks_have_voice:
        avg_score = 0.0
    elif non_voice_scores:
        # Get the maximum score from all chunks instead of clustering
        avg_score = max(non_voice_scores)
    else:
        avg_score = 0.0
    return avg_score

# === BATCH PROCESSING AND CSV OUTPUT ===
def process_directory_and_save_csv(audio_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    
    # Define standard CSV columns
    csv_columns = [
        "resource_id", "audio_file", "video_file", 
        "avg_audio_deepfake_score", "avg_voice_deepfake_score", 
        "avg_video_deepfake_score", "avg_face_deepfake_score",
        "is_audio_deepfake", "is_video_deepfake"
    ]
    
    # Load existing CSV or create new one
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "deepfake_score.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure all required columns exist
        for col in csv_columns:
            if col not in df.columns:
                df[col] = ""
        # Reorder columns to match standard format
        df = df[csv_columns]
    else:
        df = pd.DataFrame(columns=csv_columns)
    
    video_dir = os.path.join(script_dir, "video")
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        print(f"Processing {audio_file}...")
        avg_score = process_audio_for_deepfake_in_memory(audio_path, model, device, chunk_length=10)
        print(f"Average score: {avg_score:.4f}")
        
        # Check if corresponding video file exists
        video_file_candidate = os.path.splitext(audio_file)[0] + ".mp4"
        video_file_path = os.path.join(video_dir, video_file_candidate)
        video_file = video_file_candidate if os.path.exists(video_file_path) else ""
        
        # Get base name without extension for matching
        audio_base_name = os.path.splitext(audio_file)[0]
        
        # Check if audio_file already exists in CSV
        audio_match = df["audio_file"] == audio_file
        
        # Check if video_file with same base name exists in CSV
        video_match = df["video_file"].apply(lambda x: os.path.splitext(x)[0] if isinstance(x, str) and x else "") == audio_base_name
        
        pred_score = round(avg_score, 4)
        pred_label = "1" if pred_score > 0.52 else "0"
        is_audio_deepfake = 1 if pred_score > 0.52 else 0
        
        if audio_match.any():
            # Update existing row by audio_file
            df.loc[audio_match, "avg_audio_deepfake_score"] = pred_score
            df.loc[audio_match, "avg_voice_deepfake_score"] = 0
            df.loc[audio_match, "is_audio_deepfake"] = is_audio_deepfake
            if video_file:
                df.loc[audio_match, "video_file"] = video_file
        elif video_match.any():
            # Update existing row by video_file base name match
            df.loc[video_match, "avg_audio_deepfake_score"] = pred_score
            df.loc[video_match, "audio_file"] = audio_file
            df.loc[video_match, "is_audio_deepfake"] = is_audio_deepfake
        else:
            # Add new row with new resource_id
            existing_ids = df["resource_id"].dropna().tolist()
            next_id = 1
            if existing_ids:
                nums = [int(str(i).replace("vid_", "").replace("resource_", "")) for i in existing_ids if (str(i).startswith("vid_") or str(i).startswith("resource_")) and str(i).replace("vid_", "").replace("resource_", "").isdigit()]
                if nums:
                    next_id = max(nums) + 1
            new_resource_id = f"resource_{next_id:02d}"
            new_row = {col: "" for col in csv_columns}
            new_row["resource_id"] = new_resource_id
            new_row["audio_file"] = audio_file
            new_row["video_file"] = video_file
            new_row["avg_audio_deepfake_score"] = pred_score
            new_row["avg_voice_deepfake_score"] = 0
            new_row["avg_video_deepfake_score"] = 0
            new_row["avg_face_deepfake_score"] = 0
            new_row["is_audio_deepfake"] = is_audio_deepfake
            new_row["is_video_deepfake"] = 0
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {len(audio_files)} audio files processed")

# === SCRIPT ENTRYPOINT ===
if __name__ == "__main__":
    # Hardcoded paths

    process_directory_and_save_csv(audio_dir)

