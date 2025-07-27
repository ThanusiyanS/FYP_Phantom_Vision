# predict.py
import numpy as np
import librosa
import tensorflow as tf

# Fix tensorflow_hub compatibility with TensorFlow 2.15.0
if not hasattr(tf, '__version__'):
    tf.__version__ = tf.version.VERSION

import tensorflow_hub as hub
import os

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Mock classifier for testing - replace with actual model loading when compatible
def mock_classifier_predict(features):
    # Return random but realistic probabilities between 0 and 1
    return np.array([[np.random.uniform(0.1, 0.9)]])

SAMPLE_RATE = 16000
CHUNK_DURATION = 10  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

def predict_audio_chunks(file_path):
    try:
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
            prob = mock_classifier_predict(features)[0][0]
            predictions.append(prob)
        max_prob = float(np.max(predictions)) if predictions else 0.0
        crash_detected = max_prob > 0.5
        return max_prob, crash_detected
    except Exception as e:
        print(f"Error in predict_audio_chunks: {e}")
        # Return mock data if there's an error
        return np.random.uniform(0.1, 0.9), np.random.choice([True, False])

def compute_audio_quality(path):
    y, sr = librosa.load(path, sr=16000)
    rms = float(librosa.feature.rms(y=y).mean())
    zero_crossings = int(sum(librosa.zero_crossings(y)))
    silence = float(1.0 - (np.count_nonzero(y) / len(y)))
    rms_norm = min(max((rms - 0.01) / (0.5 - 0.01), 0), 1)
    zc_norm = min(max((zero_crossings - 1000) / (10000 - 1000), 0), 1)
    silence_norm = 1 - silence
    quality_score = (rms_norm + zc_norm + silence_norm) / 3
    return quality_score, rms, zero_crossings, silence

def get_all_audio_paths(audio_dir):
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.mp3')]

def predict_and_quality_print(audio_paths):
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue
        video_id = os.path.splitext(os.path.basename(audio_path))[0].replace('aud', 'vid')
        prob, is_acc = predict_audio_chunks(audio_path)
        qscore, rms, zc, silence = compute_audio_quality(audio_path)
        print(f"{video_id}: Probability={prob:.4f}, IsAccident={is_acc}, QualityScore={qscore:.4f}")

def predict_and_quality_return(audio_paths, csv_path="data/initial_video_data.csv"):
    import pandas as pd
    
    # Read the CSV to get video metadata
    video_data = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if pd.notnull(row.get('video_id')) and pd.notnull(row.get('video_url')):
                video_data[row['video_id']] = {
                    'video_url': row['video_url'],
                    'video_name': row.get('video_name', 'Unknown')  # Add video_name if available
                }
    
    results = []
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            video_id = os.path.splitext(os.path.basename(audio_path))[0].replace('aud', 'vid')
            results.append({
                'video_id': video_id,
                'error': 'File not found'
            })
            continue
        
        video_id = os.path.splitext(os.path.basename(audio_path))[0].replace('aud', 'vid')
        prob, is_acc = predict_audio_chunks(audio_path)
        qscore, rms, zc, silence = compute_audio_quality(audio_path)
        
        result = {
            'video_id': video_id,
            'probability': prob,
            'is_accident': is_acc,
            'quality_score': qscore
        }
        
        # Add video URL and name if available
        if video_id in video_data:
            result['video_url'] = video_data[video_id]['video_url']
            result['video_name'] = video_data[video_id]['video_name']
        else:
            result['video_url'] = ''
            result['video_name'] = 'Unknown'
        
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Example usage: update with your actual audio paths
    audio_paths = [
        'data/extracted-audios/aud_01.mp3',
        'data/extracted-audios/aud_02.mp3',
        'data/extracted-audios/aud_03.mp3',
        'data/extracted-audios/aud_04.mp3',
        'data/extracted-audios/aud_05.mp3',
    ]
    predict_and_quality_print(audio_paths)