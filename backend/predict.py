# predict.py
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
classifier = load_model('crash_classifier_v1.keras',compile=False)

def preprocess_audio(audio_path, desired_sr=16000):
    # Use librosa to load any audio file and resample as needed
    waveform, sr = librosa.load(audio_path, sr=desired_sr, mono=True)
    return waveform

def predict_audio(audio_path):
    # Preprocess: get waveform, always 16kHz mono float32
    waveform = preprocess_audio(audio_path)
    # Make sure it's float32 numpy
    waveform = waveform.astype(np.float32)
    # Extract embedding
    scores, embeddings, spectrogram = yamnet_model(waveform)
    embedding = tf.reduce_mean(embeddings, axis=0)  # Average over frames
    # Predict with classifier
    pred = classifier.predict(np.expand_dims(embedding, axis=0))
    return pred[0][0]  # Probability


def predict_audio_batch(audio_paths):
    results = []
    for path in audio_paths:
        prob = predict_audio(path)
        results.append({
            "path": path,
            "probability": float(prob),
            "is_accident": float(prob) > 0.5
        })
    return results
    
# # Example usage:
# test_audio_path = 'test_crash_aud_02.mp3'
# prob = predict_audio(test_audio_path)
# print(f"Probability of crash: {prob:.4f}")
# if prob > 0.5:
#     print("Prediction: CRASH")
# else:
#     print("Prediction: NON-CRASH")
