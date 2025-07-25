import os
import numpy as np
import librosa
import pandas as pd
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import torch

# === CONFIG ===
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_score.csv")
CHUNK_LENGTH = 10  # seconds

# === Load HuggingFace Model ===
checkpoint = "MelodyMachine/Deepfake-audio-detection-V2"
feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
model = AutoModelForAudioClassification.from_pretrained(checkpoint)
device = "cpu"
model.to(device)
classifier = pipeline(
    "audio-classification",
    model=model,
    feature_extractor=feature_extractor,
    device=-1
)

def chunk_audio_in_memory(audio_path, chunk_length=10):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    chunk_samples = int(sr * chunk_length)
    chunks = []
    if len(y) < chunk_samples:
        chunks.append(y)
    else:
        for i in range(0, len(y), chunk_samples):
            chunks.append(y[i:i + chunk_samples])
    return chunks, sr

def predict_chunk(chunk, sr):
    results = classifier({"array": chunk, "sampling_rate": sr})
    for r in results:
        if r["label"].lower() in ["fake", "real"]:
            return r["score"] if r["label"].lower() == "fake" else (1 - r["score"])
    return 0.0

def get_largest_cluster_mean(scores, n_clusters=2):
    from sklearn.cluster import KMeans
    X = np.array(scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    cluster_scores = X[labels == largest_cluster].flatten()
    return float(np.mean(cluster_scores))

# === MAIN LOGIC ===
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]

# Load or initialize DataFrame
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=[
        "resource_id", "audio_file", "video_file", "avg_audio_deepfake_score", "avg_car_crash_deepfake_score", "avg_video_deepfake_score"
    ])

for audio_file in audio_files:
    audio_path = os.path.join(AUDIO_DIR, audio_file)
    # --- Chunk audio and predict each chunk ---
    chunks, sr = chunk_audio_in_memory(audio_path, chunk_length=CHUNK_LENGTH)
    chunk_scores = [predict_chunk(chunk, sr) for chunk in chunks]
    if len(chunk_scores) == 1:
        avg_score = chunk_scores[0]
    else:
        avg_score = get_largest_cluster_mean(chunk_scores, n_clusters=2)
    match = df["audio_file"] == audio_file
    if match.any():
        idx = df.index[match][0]
        df.at[idx, "avg_car_crash_deepfake_score"] = round(avg_score, 4)
    else:
        new_resource_id = f"id_{len(df)+1:02d}"
        new_row = {
            "resource_id": new_resource_id,
            "audio_file": audio_file,
            "video_file": "",
            "avg_audio_deepfake_score": "",
            "avg_car_crash_deepfake_score": round(avg_score, 4),
            "avg_video_deepfake_score": ""
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

fieldnames = [
    "resource_id", "audio_file", "video_file", "avg_audio_deepfake_score", "avg_car_crash_deepfake_score", "avg_video_deepfake_score"
]
df = df[fieldnames]
df.to_csv(CSV_PATH, index=False) 


