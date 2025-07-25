# predict.py
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv
import os

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
classifier = load_model('audio_module/models/crash_classifier_audio_final_v3.keras', compile=False)

SAMPLE_RATE = 16000
CHUNK_DURATION = 10  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Predict crash probability and is_accident for an audio file

def predict_audio_chunks(file_path):
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(waveform)
    predictions = []
    for start in range(0, total_samples, CHUNK_SIZE):
        end = start + CHUNK_SIZE
        chunk = waveform[start:end]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
        _, embeddings, _ = yamnet_model(chunk)
        features = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
        prob = classifier.predict(features)[0][0]
        predictions.append(prob)
    max_prob = float(np.max(predictions)) if predictions else 0.0
    crash_detected = max_prob > 0.5
    return max_prob, crash_detected

def compute_audio_quality(path):
    y, sr = librosa.load(path, sr=16000)
    rms = float(librosa.feature.rms(y=y).mean())
    zero_crossings = int(sum(librosa.zero_crossings(y)))
    silence = float(1.0 - (np.count_nonzero(y) / len(y)))
    # Normalize each metric (example min/max values, adjust as needed)
    # These min/max values should be set based on your dataset for best results
    rms_norm = min(max((rms - 0.01) / (0.5 - 0.01), 0), 1)  # Example: 0.01-0.5
    zc_norm = min(max((zero_crossings - 1000) / (10000 - 1000), 0), 1)  # Example: 1000-10000
    silence_norm = 1 - silence  # Less silence is better
    # Aggregate as mean of normalized metrics
    quality_score = (rms_norm + zc_norm + silence_norm) / 3
    return quality_score, rms, zero_crossings, silence

def generate_audio_module_csv(input_csv_path, output_csv_path):
    import pandas as pd
    df = pd.read_csv(input_csv_path)
    # Assume audio_id can be inferred from audio_path (e.g., aud_01.mp3)
    if 'audio_path' in df.columns:
        df['audio_id'] = df['audio_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0] if pd.notnull(x) else '')
    # Use probability as probability_score
    if 'probability' in df.columns:
        df['probability_score'] = df['probability']
    else:
        df['probability_score'] = None
    # Select and rename columns
    final_df = df[['video_id', 'audio_id', 'probability_score', 'quality_score', 'is_accident']]
    final_df.to_csv(output_csv_path, index=False)
    return final_df

# Batch function: takes a list of audio paths, updates CSV with prediction and quality

def predict_and_quality_batch(audio_paths, csv_path):
    print('predicting and quality assessing')
    # Read existing CSV rows
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with open(csv_path, 'r', newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))
    # Update each row with prediction and quality
    for row in reader:
        audio_path = None
        # Try to infer audio path from video_path
        if row.get('video_path'):
            base = os.path.splitext(os.path.basename(row['video_path']))[0]
            audio_path = os.path.join('backend/data/extracted-audios', base + '.mp3')
        if audio_path and os.path.exists(audio_path):
            prob, is_acc = predict_audio_chunks(audio_path)
            qscore, rms, zc, silence = compute_audio_quality(audio_path)
            row['probability'] = prob
            row['is_accident'] = is_acc
            row['quality_score'] = qscore
            row['rms'] = rms
            row['zero_crossings'] = zc
            row['silence'] = silence
    # Write back to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "video_url", "video_path", "probability", "is_accident", "quality_score", "rms", "zero_crossings", "silence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
    # Generate audio_module.csv
    audio_module_csv_path = os.path.join(os.path.dirname(csv_path), 'audio_module.csv')
    generate_audio_module_csv(csv_path, audio_module_csv_path)
    return reader
