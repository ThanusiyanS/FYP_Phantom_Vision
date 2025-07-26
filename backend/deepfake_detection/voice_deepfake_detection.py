import os
import numpy as np
import librosa
import pandas as pd
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import webrtcvad

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

# === webrtcvad-based Voice Activity Detection ===
def is_human_voice_chunk_webrtcvad(chunk, sr, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    # Convert to 16-bit PCM
    if chunk.dtype != np.int16:
        chunk_pcm = (chunk * 32767).astype(np.int16)
    else:
        chunk_pcm = chunk
    frame_duration = 30  # ms
    frame_length = int(sr * frame_duration / 1000)
    n_frames = len(chunk_pcm) // frame_length
    for i in range(n_frames):
        frame = chunk_pcm[i * frame_length : (i + 1) * frame_length]
        if len(frame) < frame_length:
            continue
        if vad.is_speech(frame.tobytes(), sample_rate=sr):
            return True
    return False

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
        if r["label"].lower() == "fake":
            return 1.0 - float(r["score"])
    for r in results:
        if r["label"].lower() == "real":
            return float(r["score"])
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

# Define all required columns
csv_columns = [
    "resource_id",
    "audio_file", 
    "video_file", 
    "avg_audio_deepfake_score", 
    "avg_voice_deepfake_score", 
    "avg_video_deepfake_score",
    "avg_face_deepfake_score",
    "deepfake_prediction_score",
    "deepfake_prediction_label"
]

# Load or initialize DataFrame
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    # Ensure required columns exist
    for col in csv_columns:
        if col not in df.columns:
            df[col] = ""
    # Reorder columns if necessary
    df = df[csv_columns]
else:
    df = pd.DataFrame(columns=csv_columns)

for audio_file in audio_files:
    audio_path = os.path.join(AUDIO_DIR, audio_file)
    # --- Chunk audio and predict each chunk ---
    chunks, sr = chunk_audio_in_memory(audio_path, chunk_length=CHUNK_LENGTH)
    chunk_scores = [predict_chunk(chunk, sr) for chunk in chunks]
    if len(chunk_scores) == 1:
        avg_score = chunk_scores[0]
    else:
        avg_score = get_largest_cluster_mean(chunk_scores, n_clusters=2)
    pred_score = round(avg_score, 4)
    pred_label = ""
    if audio_file in df["audio_file"].values:
        # Update existing row
        df.loc[df["audio_file"] == audio_file, "avg_voice_deepfake_score"] = pred_score
        df.loc[df["audio_file"] == audio_file, "deepfake_prediction_score"] = pred_score
        df.loc[df["audio_file"] == audio_file, "deepfake_prediction_label"] = pred_label
    else:
        # Generate new resource_id
        existing_ids = df["resource_id"].dropna().tolist()
        next_id = 1
        if existing_ids:
            # Extract numeric part and increment
            nums = [int(str(i).replace("id_", "").replace("resource_", "")) for i in existing_ids if (str(i).startswith("id_") or str(i).startswith("resource_")) and str(i).replace("id_", "").replace("resource_", "").isdigit()]
            if nums:
                next_id = max(nums) + 1
        new_resource_id = f"resource_{next_id:02d}"
        new_row = {col: "" for col in csv_columns}
        new_row["resource_id"] = new_resource_id
        new_row["audio_file"] = audio_file
        new_row["avg_voice_deepfake_score"] = pred_score
        new_row["deepfake_prediction_score"] = pred_score
        new_row["deepfake_prediction_label"] = pred_label
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(CSV_PATH, index=False) 


