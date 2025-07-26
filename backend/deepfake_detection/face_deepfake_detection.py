import torch
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
import os
from sklearn.cluster import KMeans
from scipy import stats

# Add icpr2020dfdc to path
sys.path.append('icpr2020dfdc/icpr2020dfdc')

from architectures import fornet, weights
from blazeface import FaceExtractor, BlazeFace, VideoReader
from isplutils import utils

# --- CONFIG ---
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
face_policy = 'scale'
face_size = 224
frames_per_video = 96  # Increased for better sampling
face_confidence_threshold = 0.7  # Increased threshold for better quality faces
min_face_size = 50  # Minimum face size in pixels
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_score.csv")

# --- Load model ---
model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(torch.hub.load_state_dict_from_url(model_url, map_location=device, check_hash=True))

# Validate model loading
print(f"Model loaded successfully. Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

# --- Preprocessing ---
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

# --- Face detector and extractor ---
facedet = BlazeFace().to(device)
facedet.load_weights("icpr2020dfdc/blazeface/blazeface.pth")
facedet.load_anchors("icpr2020dfdc/blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

def get_largest_cluster_mean(scores, n_clusters=2):
    """Get the mean of the largest cluster of scores"""
    if len(scores) < n_clusters:
        return np.mean(scores)
    
    scores_reshaped = np.array(scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(n_clusters, len(scores)), random_state=42)
    kmeans.fit(scores_reshaped)
    
    # Find the largest cluster
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    
    # Get scores in the largest cluster
    largest_cluster_scores = scores[kmeans.labels_ == largest_cluster_idx]
    return np.mean(largest_cluster_scores)

def filter_outliers(scores, threshold=2.0):
    """Remove statistical outliers using z-score"""
    if len(scores) < 3:
        return scores
    
    # Convert to numpy array if it's not already
    scores_array = np.array(scores)
    z_scores = np.abs(stats.zscore(scores_array))
    return scores_array[z_scores < threshold]

def normalize_predictions(preds, clip_range=5.0):
    """Normalize and clip predictions to reasonable range"""
    return np.clip(preds, -clip_range, clip_range)

def get_face_quality_score(face_info):
    """Calculate face quality score based on confidence and size"""
    confidence = face_info['confidence']
    bbox = face_info['bbox']
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size = min(width, height)
    
    # Normalize size (assuming typical face size is around 100-200 pixels)
    size_score = min(size / 150.0, 1.0)
    
    # Combined quality score
    quality_score = confidence * size_score
    return quality_score

def check_temporal_consistency(scores, window_size=5, threshold=0.3):
    """Check for temporal consistency in predictions"""
    if len(scores) < window_size:
        return scores
    
    consistent_scores = []
    for i in range(len(scores)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(scores), i + window_size // 2 + 1)
        window_scores = scores[start_idx:end_idx]
        
        # Check if current score is consistent with window
        window_mean = np.mean(window_scores)
        if abs(scores[i] - window_mean) < threshold:
            consistent_scores.append(scores[i])
    
    return consistent_scores if consistent_scores else scores

def get_robust_score(scores, confidence_threshold=0.1):
    """Get a more robust score using multiple methods"""
    if len(scores) == 0:
        return 0.0
    
    # Convert to list if it's a numpy array
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    # Simple approach: use the mean of the top 70% of scores
    # This preserves high scores for fake videos while filtering some noise
    sorted_scores = np.sort(scores_list)
    n = len(sorted_scores)
    top_percentile = 0.7  # Use top 70% of scores
    start_idx = int(n * (1 - top_percentile))
    
    if start_idx < n:
        robust_scores = sorted_scores[start_idx:]
        final_score = np.mean(robust_scores)
    else:
        final_score = np.mean(scores_list)
    
    return float(final_score)

def get_video_chunks(video_path, chunk_length_sec=10):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunk_size = int(fps * chunk_length_sec)
    chunks = []
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        frame_idxs = list(range(start, end))
        if len(frame_idxs) == 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for idx in frame_idxs:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if frames:
            chunks.append(np.stack(frames))
    cap.release()
    return chunks

# --- Deepfake prediction function ---
def predict_video_deepfake(video_path):
    """Predict deepfake score for a video"""
    print(f"Video: {os.path.basename(video_path)}")
    
    try:
        # Extract faces from video
        faces = face_extractor.process_video(video_path)
        
        if not faces:
            print("  No faces detected in video")
            return 0.0
        
        # Debug: Print the structure of faces
        print(f"  Faces structure type: {type(faces)}")
        print(f"  Total frames: {len(faces)}")
        
        if len(faces) > 0:
            print(f"  First frame type: {type(faces[0])}")
            if isinstance(faces[0], dict):
                print(f"  First frame dict keys: {list(faces[0].keys())}")
                if 'faces' in faces[0]:
                    print(f"  First frame 'faces' type: {type(faces[0]['faces'])}")
                    if hasattr(faces[0]['faces'], '__len__'):
                        print(f"  First frame 'faces' length: {len(faces[0]['faces'])}")
                if 'scores' in faces[0]:
                    print(f"  First frame 'scores' type: {type(faces[0]['scores'])}")
                    if hasattr(faces[0]['scores'], '__len__'):
                        print(f"  First frame 'scores' length: {len(faces[0]['scores'])}")
            elif hasattr(faces[0], '__len__'):
                print(f"  First frame length: {len(faces[0])}")
                if len(faces[0]) > 0:
                    print(f"  First face type: {type(faces[0][0])}")
        
        # Get high-quality faces only
        face_confidence_threshold = 0.7
        min_face_size = 50
        high_quality_faces = []
        
        # Handle different possible face data structures
        for frame_idx, frame_faces in enumerate(faces):
            try:
                if isinstance(frame_faces, dict):
                    # If frame_faces is a dict with 'faces' key
                    if 'faces' in frame_faces and 'scores' in frame_faces:
                        faces_list = frame_faces['faces']
                        scores_list = frame_faces['scores']
                        
                        # Debug: Print lengths for first frame
                        if frame_idx == 0:
                            print(f"  Faces list length: {len(faces_list)}")
                            print(f"  Scores list length: {len(scores_list)}")
                        
                        # Make sure we have matching lengths
                        min_length = min(len(faces_list), len(scores_list))
                        for face_idx in range(min_length):
                            face_img = faces_list[face_idx]
                            score = scores_list[face_idx]
                            
                            if score >= face_confidence_threshold:
                                # Check face size
                                if face_img.shape[0] >= min_face_size and face_img.shape[1] >= min_face_size:
                                    high_quality_faces.append({
                                        'face': face_img,
                                        'confidence': score,
                                        'bbox': [0, 0, face_img.shape[1], face_img.shape[0]]  # Approximate bbox
                                    })
                elif isinstance(frame_faces, list):
                    # If frame_faces is a list of face images
                    for face_img in frame_faces:
                        if isinstance(face_img, np.ndarray):
                            if face_img.shape[0] >= min_face_size and face_img.shape[1] >= min_face_size:
                                high_quality_faces.append({
                                    'face': face_img,
                                    'confidence': 1.0,  # Default confidence
                                    'bbox': [0, 0, face_img.shape[1], face_img.shape[0]]
                                })
                else:
                    # Try to handle as face images directly
                    if isinstance(frame_faces, np.ndarray):
                        if frame_faces.shape[0] >= min_face_size and frame_faces.shape[1] >= min_face_size:
                            high_quality_faces.append({
                                'face': frame_faces,
                                'confidence': 1.0,
                                'bbox': [0, 0, frame_faces.shape[1], frame_faces.shape[0]]
                            })
            except Exception as e:
                print(f"  Error processing frame {frame_idx}: {str(e)}")
                continue
        
        if not high_quality_faces:
            print("  No high-quality faces detected")
            return 0.0
        
        # Process faces and get predictions
        raw_predictions = []
        quality_weights = []
        
        for face_info in high_quality_faces:
            try:
                face_img = face_info['face']
                face_tensor = transf(image=face_img)['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = net(face_tensor).cpu().numpy().flatten()[0]
                
                # Store raw prediction (before sigmoid)
                raw_predictions.append(pred)
                
                # Calculate face quality weight
                quality_weight = get_face_quality_score(face_info)
                quality_weights.append(quality_weight)
            except Exception as e:
                print(f"  Error processing face: {str(e)}")
                continue
        
        # Convert to numpy arrays
        raw_predictions = np.array(raw_predictions)
        quality_weights = np.array(quality_weights)
        
        # Print debug information
        print(f"  Total frames processed: {frames_per_video}")
        print(f"  Frames with detected faces: {len(faces)}")
        print(f"  Face detection rate: {len(faces)/frames_per_video*100:.1f}%")
        print(f"  High quality faces (>= {face_confidence_threshold} conf, >= {min_face_size}px): {len(high_quality_faces)}")
        print(f"  Raw predictions: min={raw_predictions.min():.4f}, max={raw_predictions.max():.4f}, mean={raw_predictions.mean():.4f}")
        print(f"  Number of face predictions: {len(raw_predictions)}")
        
        # Calculate deepfake score
        if len(raw_predictions) > 0:
            # Normalize weights
            quality_weights = quality_weights / np.sum(quality_weights)
            
            # Use normal sigmoid (correct interpretation)
            sigmoid_scores = torch.sigmoid(torch.tensor(raw_predictions)).numpy()
            
            # Weighted mean score (no robust filtering)
            final_score = float(np.sum(sigmoid_scores * quality_weights) / np.sum(quality_weights))
            print(f"  Weighted mean deepfake score: {final_score:.4f}")
            return final_score
        else:
            return 0.0
            
    except Exception as e:
        print(f"  Error in predict_video_deepfake: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0

# --- CSV update function ---
def update_deepfake_score_csv(video_path, avg_score):
    """Update the deepfake_score.csv file with the new score, using resource_id, video_file, and avg_face_deepfake_score columns. Always create the CSV with all required columns."""
    video_file = os.path.basename(video_path)
    csv_columns = [
        "resource_id", "audio_file", "video_file", "avg_audio_deepfake_score", "avg_voice_deepfake_score", "avg_video_deepfake_score", "avg_face_deepfake_score", "deepfake_prediction_score", "deepfake_prediction_label"
    ]
    
    # Load existing CSV or create new one
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
    
    # Check if video already exists
    if video_file in df["video_file"].values:
        # Update existing row
        pred_score = round(avg_score, 4)
        pred_label = ""
        df.loc[df["video_file"] == video_file, "avg_face_deepfake_score"] = avg_score
        df.loc[df["video_file"] == video_file, "deepfake_prediction_score"] = pred_score
        df.loc[df["video_file"] == video_file, "deepfake_prediction_label"] = pred_label
    else:
        # Generate new resource_id
        existing_ids = df["resource_id"].dropna().tolist()
        next_id = 1
        if existing_ids:
            # Extract numeric part and increment
            nums = [int(str(i).replace("vid_", "").replace("resource_", "")) for i in existing_ids if (str(i).startswith("vid_") or str(i).startswith("resource_")) and str(i).replace("vid_", "").replace("resource_", "").isdigit()]
            if nums:
                next_id = max(nums) + 1
        new_resource_id = f"resource_{next_id:02d}"
        new_row = {col: "" for col in csv_columns}
        new_row["resource_id"] = new_resource_id
        new_row["video_file"] = video_file
        new_row["avg_face_deepfake_score"] = avg_score
        new_row["deepfake_prediction_score"] = pred_score
        new_row["deepfake_prediction_label"] = pred_label
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"Updated {CSV_PATH} with avg_face_deepfake_score: {avg_score:.4f} for {video_file}")

# --- Example usage ---
if __name__ == "__main__":
    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
    
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        exit(1)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("No video files found in the video directory")
        exit(1)
    
    print(f"Found {len(video_files)} video files to process")
    print("=" * 50)
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        try:
            avg_score = predict_video_deepfake(video_path)
            update_deepfake_score_csv(video_path, avg_score)
            print(f"Final average deepfake score for {video_file}: {avg_score:.4f}")
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
        print("-" * 50)
