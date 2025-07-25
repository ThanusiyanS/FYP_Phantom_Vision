import os
import torch
import cv2
import numpy as np
import pandas as pd

# --- CONFIG ---
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_score.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_video_model_v4.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 16

# --- DUMMY MODEL LOADING (replace with your actual model) ---
class DummyVideoModel(torch.nn.Module):
    def forward(self, x):
        # Replace with your actual model logic
        return torch.sigmoid(torch.rand(1))

# model = torch.load(MODEL_PATH, map_location=DEVICE)
model = DummyVideoModel().to(DEVICE)
model.eval()

# --- FRAME TRANSFORM (replace with your actual transform) ---
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_largest_cluster_mean(scores, n_clusters=2):
    from sklearn.cluster import KMeans
    X = np.array(scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    cluster_scores = X[labels == largest_cluster].flatten()
    return float(np.mean(cluster_scores))

def predict_deepfake(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Optionally: clean overlays/logos here
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(rgb_frame))
    cap.release()
    if len(frames) < max_frames:
        return None  # Skip short clips
    # Predict per-frame score
    frame_scores = []
    with torch.no_grad():
        for frame_tensor in frames:
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE) if len(frame_tensor.shape) == 3 else frame_tensor.unsqueeze(0).to(DEVICE)
            prob = model(frame_tensor).item()
            frame_scores.append(prob)
    if len(frame_scores) == 1:
        avg_score = frame_scores[0]
    else:
        avg_score = get_largest_cluster_mean(frame_scores, n_clusters=2)
    return avg_score

# --- PROCESS ALL VIDEOS ---
results = []
for idx, video_file in enumerate(sorted(os.listdir(VIDEO_DIR)), 1):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        score = predict_deepfake(video_path)
        if score is not None:
            results.append({
                "video_file": video_file,
                "avg_video_deepfake_score": round(score, 4)
            })

# --- UPDATE OR CREATE CSV ---
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    video_df = pd.DataFrame(results)
    # Remove file extensions for matching
    df["_audio_base"] = df["audio_file"].apply(lambda x: os.path.splitext(x)[0])
    video_df["_video_base"] = video_df["video_file"].apply(lambda x: os.path.splitext(x)[0])
    # Update matching rows
    for i, vrow in video_df.iterrows():
        match = df["_audio_base"] == vrow["_video_base"]
        if match.any():
            df.loc[match, "avg_video_deepfake_score"] = vrow["avg_video_deepfake_score"]
        else:
            # Add as new row with new resource_id
            new_resource_id = f"vid_{len(df)+1:02d}"
            new_row = {
                "resource_id": new_resource_id,
                "audio_file": "",
                "video_file": vrow["video_file"],
                "avg_audio_deepfake_score": "",
                "avg_video_deepfake_score": vrow["avg_video_deepfake_score"]
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.drop(columns=["_audio_base"], errors="ignore")
    df = df.drop(columns=["_video_base"], errors="ignore")
    df.to_csv(CSV_PATH, index=False)
else:
    # If no CSV exists, just write the video results with new resource_ids
    video_df = pd.DataFrame(results)
    video_df["resource_id"] = [f"vid_{i+1:02d}" for i in range(len(video_df))]
    video_df["audio_file"] = ""
    video_df["avg_audio_deepfake_score"] = ""
    video_df = video_df[["resource_id", "audio_file", "video_file", "avg_audio_deepfake_score", "avg_video_deepfake_score"]]
    video_df.to_csv(CSV_PATH, index=False)