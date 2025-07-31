import os
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.cluster import KMeans

# Optional import for OCR functionality
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not available. OCR-based text removal will be skipped.")

# --- CONFIG ---
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","data","retrieved-videos")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_score.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_video_model_v4.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOGO/WATERMARK REMOVAL FUNCTIONS ---
def remove_text_regions(frame, method='inpaint'):
    """Detect text with OCR and mask it via inpainting, blur, or solid fill."""
    if not OCR_AVAILABLE:
        return frame  # Skip text removal if OCR not available
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        mask = np.zeros_like(gray)

        for i in range(len(boxes['text'])):
            text = boxes['text'][i]
            if len(text.strip()) > 0:
                x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        if method == 'inpaint':
            cleaned = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        elif method == 'blur':
            cleaned = frame.copy()
            cleaned[mask == 255] = cv2.GaussianBlur(cleaned, (15, 15), 0)[mask == 255]
        elif method == 'black':
            cleaned = frame.copy()
            cleaned[mask == 255] = 0
        else:
            cleaned = frame

        return cleaned
    except Exception as e:
        # If OCR fails, return original frame
        print(f"OCR processing failed: {e}")
        return frame

def remove_logos_and_watermarks(frame):
    """Remove logos, watermarks, and unwanted PNGs from video frame"""
    # Method 1: Text removal using OCR (if available)
    if OCR_AVAILABLE:
        frame = remove_text_regions(frame, method='inpaint')
    
    # Method 2: Logo detection using contour analysis (always available)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find potential logo regions
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for logo regions
    logo_mask = np.zeros_like(gray)
    
    for contour in contours:
        # Filter contours by area (remove very small and very large)
        area = cv2.contourArea(contour)
        if 100 < area < 10000:  # Adjust these thresholds as needed
            x, y, w, h = cv2.boundingRect(contour)
            # Check aspect ratio to identify potential logos
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio for logos
                cv2.rectangle(logo_mask, (x, y), (x + w, y + h), 255, -1)
    
    # Apply inpainting to remove detected logos
    if np.sum(logo_mask) > 0:
        frame = cv2.inpaint(frame, logo_mask, 3, cv2.INPAINT_TELEA)
    
    return frame

# --- MODEL ARCHITECTURE ---
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)
        _, (h, _) = self.lstm(feats)
        out = self.fc(h[-1])
        return torch.sigmoid(out).squeeze()

# --- MODEL LOADING ---
model = CNN_LSTM().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- CLUSTERING FUNCTION ---
def get_largest_cluster_mean(scores, n_clusters=2):
    if len(scores) < n_clusters:
        return float(np.mean(scores))
    
    X = np.array(scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(n_clusters, len(scores)), random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    cluster_scores = X[labels == largest_cluster].flatten()
    return float(np.mean(cluster_scores))

# --- VIDEO CHUNKING AND PREDICTION ---
def predict_deepfake(video_path, chunk_duration=10):
    """
    Split video into 10-second chunks and predict deepfake score for each chunk
    For videos shorter than 10 seconds, treat the entire video as one chunk
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    chunk_scores = []
    
    if duration < chunk_duration:
        # For videos shorter than 10 seconds, treat entire video as one chunk
        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Remove logos, watermarks, and unwanted elements
            cleaned_frame = remove_logos_and_watermarks(frame)
            
            # Convert BGR to RGB and apply transform
            rgb_frame = cv2.cvtColor(cleaned_frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(rgb_frame))
        
        # Skip if insufficient frames
        if len(frames) < 16:  # Minimum frames for model
            cap.release()
            return None
        
        # Stack frames and add batch dimension
        frame_tensor = torch.stack(frames)  # [seq_len, c, h, w]
        frame_tensor = frame_tensor.unsqueeze(0)  # [1, seq_len, c, h, w]
        
        # Predict score for this single chunk
        with torch.no_grad():
            frame_tensor = frame_tensor.to(DEVICE)
            prob = model(frame_tensor).item()
            chunk_scores.append(prob)
    else:
        # For videos 10 seconds or longer, split into chunks
        frames_per_chunk = int(fps * chunk_duration)
        
        for chunk_start in range(0, total_frames, frames_per_chunk):
            chunk_end = min(chunk_start + frames_per_chunk, total_frames)
            frames = []
            
            # Set frame position to start of chunk
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
            
            # Read frames for this chunk
            for _ in range(chunk_end - chunk_start):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Remove logos, watermarks, and unwanted elements
                cleaned_frame = remove_logos_and_watermarks(frame)
                
                # Convert BGR to RGB and apply transform
                rgb_frame = cv2.cvtColor(cleaned_frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(rgb_frame))
            
            # Skip chunks with insufficient frames
            if len(frames) < 16:  # Minimum frames for model
                continue
            
            # Stack frames and add batch dimension
            frame_tensor = torch.stack(frames)  # [seq_len, c, h, w]
            frame_tensor = frame_tensor.unsqueeze(0)  # [1, seq_len, c, h, w]
            
            # Predict score for this chunk
            with torch.no_grad():
                frame_tensor = frame_tensor.to(DEVICE)
                prob = model(frame_tensor).item()
                chunk_scores.append(prob)
    
    cap.release()
    
    if not chunk_scores:
        return None
    
    # Calculate average score using clustering method
    avg_score = get_largest_cluster_mean(chunk_scores)
    return avg_score

# --- PROCESS ALL VIDEOS ---
def process_video_files(video_dir, predict_func):
    """
    Process all video files in the specified directory and return results.
    
    Args:
        video_dir (str): Directory containing video files
        predict_func (callable): Function to predict deepfake scores
    
    Returns:
        list: List of dictionaries containing video file names and their scores
    """
    results = []
    for idx, video_file in enumerate(sorted(os.listdir(video_dir)), 1):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing {video_file}...")
            score = predict_func(video_path)
            if score is not None:
                results.append({
                    "video_file": video_file,
                    "avg_video_deepfake_score": round(score, 4)
                })
                print(f"Score: {score:.4f}")
            else:
                print(f"Skipped {video_file} (too short or processing failed)")
    return results

# --- UPDATE OR CREATE CSV ---
csv_columns = [
    "resource_id", "audio_file", "video_file", "avg_audio_deepfake_score", 
    "avg_voice_deepfake_score", "avg_video_deepfake_score", "avg_face_deepfake_score", 
    "is_audio_deepfake", "is_video_deepfake"
]

def calculate_prediction_metrics(score):
    """
    Calculate prediction metrics based on video deepfake score.
    
    Args:
        score (float): Video deepfake score
    
    Returns:
        tuple: (pred_score, pred_label, is_video_deepfake)
    """
    pred_score = round(score, 4) if score != "" else ""
    pred_label = "1" if pred_score != "" and pred_score > 0.65 else ("0" if pred_score != "" else "")
    is_video_deepfake = 1 if pred_score != "" and pred_score > 0.65 else 0
    return pred_score, pred_label, is_video_deepfake

def get_next_resource_id(existing_ids):
    """
    Generate the next available resource ID.
    
    Args:
        existing_ids (list): List of existing resource IDs
    
    Returns:
        str: Next available resource ID
    """
    next_id = 1
    if existing_ids:
        nums = [int(str(i).replace("vid_", "").replace("resource_", "")) 
                for i in existing_ids 
                if (str(i).startswith("vid_") or str(i).startswith("resource_")) 
                and str(i).replace("vid_", "").replace("resource_", "").isdigit()]
        if nums:
            next_id = max(nums) + 1
    return f"resource_{next_id:02d}"

def update_existing_csv(df, results, csv_columns):
    """
    Update existing CSV with new video processing results.
    
    Args:
        df (DataFrame): Existing CSV data
        results (list): List of video processing results
        csv_columns (list): List of CSV column names
    
    Returns:
        DataFrame: Updated DataFrame
    """
    video_df = pd.DataFrame(results)
    
    # Update matching rows
    for i, vrow in video_df.iterrows():
        video_file = vrow["video_file"]
        video_base_name = os.path.splitext(video_file)[0]
        
        # Check if video_file already exists in CSV
        video_match = df["video_file"] == video_file
        
        # Check if audio_file with same base name exists in CSV
        audio_match = df["audio_file"].apply(
            lambda x: os.path.splitext(x)[0] if isinstance(x, str) and x else ""
        ) == video_base_name
        
        pred_score = round(vrow["avg_video_deepfake_score"], 4) if vrow["avg_video_deepfake_score"] != "" else ""
        pred_label = "1" if pred_score != "" and pred_score > 0.68 else ("0" if pred_score != "" else "")
        is_video_deepfake = 1 if pred_score != "" and pred_score > 0.68 else 0
        
        if video_match.any():
            # Update existing row by video_file
            df.loc[video_match, "avg_video_deepfake_score"] = vrow["avg_video_deepfake_score"]
            df.loc[video_match, "is_video_deepfake"] = is_video_deepfake
        elif audio_match.any():
            # Update existing row by audio_file base name match
            df.loc[audio_match, "avg_video_deepfake_score"] = vrow["avg_video_deepfake_score"]
            df.loc[audio_match, "video_file"] = video_file
            df.loc[audio_match, "is_video_deepfake"] = is_video_deepfake
        else:
            # Add as new row with new resource_id
            existing_ids = df["resource_id"].dropna().tolist()
            new_resource_id = get_next_resource_id(existing_ids)
            new_row = {col: "" for col in csv_columns}
            new_row["resource_id"] = new_resource_id
            new_row["video_file"] = video_file
            new_row["avg_video_deepfake_score"] = vrow["avg_video_deepfake_score"]
            new_row["avg_audio_deepfake_score"] = 0
            new_row["avg_voice_deepfake_score"] = 0
            new_row["avg_face_deepfake_score"] = 0
            new_row["is_audio_deepfake"] = 0
            new_row["is_video_deepfake"] = is_video_deepfake
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df

def create_new_csv(results, csv_columns):
    """
    Create a new CSV file with video processing results.
    
    Args:
        results (list): List of video processing results
        csv_columns (list): List of CSV column names
    
    Returns:
        DataFrame: New DataFrame with video results
    """
    video_df = pd.DataFrame(results)
    video_df["resource_id"] = [f"resource_{i+1:02d}" for i in range(len(video_df))]
    video_df["audio_file"] = ""
    video_df["avg_audio_deepfake_score"] = 0
    video_df["avg_voice_deepfake_score"] = 0
    video_df["avg_face_deepfake_score"] = 0
    video_df["is_audio_deepfake"] = 0
    # Set is_video_deepfake based on video scores
    video_df["is_video_deepfake"] = video_df["avg_video_deepfake_score"].apply(lambda x: 1 if x > 0.68 else 0)
    video_df = video_df[csv_columns]
    video_df.to_csv(CSV_PATH, index=False)
    print(f"Created {CSV_PATH} with {len(results)} video files processed")