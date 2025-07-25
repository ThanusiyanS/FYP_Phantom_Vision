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
        # Cluster the scores and use the mean of the largest cluster
        if len(non_voice_scores) == 1:
            avg_score = non_voice_scores[0]
        else:
            X = np.array(non_voice_scores).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = kmeans.fit_predict(X)
            # Find the largest cluster
            unique, counts = np.unique(labels, return_counts=True)
            largest_cluster = unique[np.argmax(counts)]
            cluster_scores = X[labels == largest_cluster].flatten()
            avg_score = float(np.mean(cluster_scores))
    else:
        avg_score = 0.0
    return avg_score

# === BATCH PROCESSING AND CSV OUTPUT ===
def process_directory_and_save_csv(audio_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, "video")
    for idx, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(audio_dir, audio_file)
        avg_score = process_audio_for_deepfake_in_memory(audio_path, model, device, chunk_length=10)
        # Check if corresponding video file exists
        video_file_candidate = os.path.splitext(audio_file)[0] + ".mp4"
        video_file_path = os.path.join(video_dir, video_file_candidate)
        if os.path.exists(video_file_path):
            video_file = video_file_candidate
            avg_video_deepfake_score = ""
        else:
            video_file = ""
            avg_video_deepfake_score = ""
        results.append({
            "resource_id": f"id_{idx:02d}",
            "audio_file": audio_file,
            "video_file": video_file,
            "avg_audio_deepfake_score": round(avg_score, 4),
            "avg_car_crash_deepfake_score": round(avg_score, 4),
            "avg_video_deepfake_score": avg_video_deepfake_score
        })
    # Always save to backend/deepfake_detection/deepfake_score.csv
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(output_dir, "deepfake_score.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["resource_id", "audio_file", "video_file", "avg_audio_deepfake_score", "avg_car_crash_deepfake_score", "avg_video_deepfake_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

# === SCRIPT ENTRYPOINT ===
if __name__ == "__main__":
    # Hardcoded paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = "/Users/danurahathevanayagam/Documents/UoM/L4S1/FYP/FYP/FYP_Phantom_Vision/backend/deepfake_detection/audio"
    model_path = os.path.join(script_dir, "fine_tuned_audio_model_v3.pth")
    process_directory_and_save_csv(audio_dir, model_path)

