"""
Car Crash Video Analysis Script
Analyzes videos for accident detection using fine-tuned VideoMAE model
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration - CORRECTED PATHS
# Since we're running from the data directory, we need to adjust paths
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir,"model")
# MODEL_PATH = "model"  # Changed from "data/model"
VIDEOS_PATH = os.path.join(script_dir,"..","data","retrieved-videos")  # Changed from "data/videos-test"
OUTPUT_CSV = os.path.join(script_dir,"video_output.csv")  # Changed from "data/video_output.csv"

# Model parameters
NUM_FRAMES = 16
TARGET_SIZE = (224, 224)

# Quality thresholds
MIN_RESOLUTION = (480, 360)
MIN_FPS = 15
MAX_BLUR_THRESHOLD = 50

def extract_and_preprocess_frames(video_path, num_frames=NUM_FRAMES, target_size=TARGET_SIZE):
    """
    Extract and preprocess frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target size for frames (width, height)
    
    Returns:
        List of PIL Images
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return frames
    
    # Sample frames evenly across the video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and convert to RGB
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

def assess_video_quality(video_path):
    """
    Comprehensive video quality assessment.
    
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'resolution_width': 0,
        'resolution_height': 0,
        'fps': 0,
        'duration_seconds': 0,
        'total_frames': 0,
        'blur_score': 0,
        'quality_issues': [],
        'overall_quality': 'Unknown'
    }
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        quality_report['quality_issues'].append('Could not open video')
        quality_report['overall_quality'] = 'Error'
        return quality_report
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    quality_report['resolution_width'] = width
    quality_report['resolution_height'] = height
    quality_report['fps'] = fps
    quality_report['duration_seconds'] = duration
    quality_report['total_frames'] = frame_count
    
    # Check resolution
    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        quality_report['quality_issues'].append(f'Low resolution: {width}x{height}')
    
    # Check FPS
    if fps < MIN_FPS:
        quality_report['quality_issues'].append(f'Low frame rate: {fps:.1f} FPS')
    
    # Check blur on sample frames
    blur_scores = []
    sample_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(blur_value)
    
    if blur_scores:
        avg_blur = np.mean(blur_scores)
        quality_report['blur_score'] = avg_blur
        
        if avg_blur < MAX_BLUR_THRESHOLD:
            quality_report['quality_issues'].append(f'Blurry video: score {avg_blur:.1f}')
    
    cap.release()
    
    # Determine overall quality
    if quality_report['quality_issues']:
        quality_report['overall_quality'] = 'Poor'
    else:
        quality_report['overall_quality'] = 'Good'
    
    return quality_report

def load_model():
    """
    Load the fine-tuned VideoMAE model and processor.
    
    Returns:
        Tuple of (processor, model) or (None, None) if loading fails
    """
    try:
        print(f"Loading model from: {MODEL_PATH}")
        
        # Check if model files exist
        model_files = ['model.safetensors', 'config.json', 'preprocessor_config.json']
        missing_files = []
        for file in model_files:
            file_path = os.path.join(MODEL_PATH, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
            else:
                print(f"  ✓ Found: {file}")
        
        if missing_files:
            print(f"\n❌ Missing model files: {', '.join(missing_files)}")
            print(f"Please ensure all model files are in the '{MODEL_PATH}' directory")
            return None, None
        
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully on {device}")
        return processor, model
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def classify_video(video_path, processor, model):
    """
    Classify a video as accident or non-accident.
    
    Returns:
        Dictionary with classification results
    """
    results = {
        'prediction': 'unknown',
        'accident_probability': 0.0,
        'non_accident_probability': 0.0,
        'confidence': 0.0,
        'processing_error': None
    }
    
    try:
        # Extract frames
        frames = extract_and_preprocess_frames(video_path)
        
        if not frames:
            results['processing_error'] = 'No frames extracted'
            return results
        
        # Process frames
        inputs = processor(frames, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predictions
        probs = probabilities[0].cpu().numpy()
        prediction_idx = np.argmax(probs)
        
        results['accident_probability'] = float(probs[0])
        results['non_accident_probability'] = float(probs[1]) if len(probs) > 1 else 0.0
        results['prediction'] = 'non-accident' if prediction_idx == 1 else 'accident'
        results['confidence'] = float(np.max(probs))
        
    except Exception as e:
        results['processing_error'] = str(e)
    
    return results

def analyze_videos():
    """
    Main function to analyze all videos in the test directory.
    """
    print("\n🔍 Checking directories...")
    
    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model directory not found at '{MODEL_PATH}'")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Contents: {os.listdir('.')}")
        return
    
    if not os.path.exists(VIDEOS_PATH):
        print(f"❌ Error: Videos directory not found at '{VIDEOS_PATH}'")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Contents: {os.listdir('.')}")
        return
    
    # Load model
    print("\n📦 Loading model...")
    processor, model = load_model()
    if processor is None or model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Get list of videos
    video_files = [f for f in os.listdir(VIDEOS_PATH) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkk'))]
    
    if not video_files:
        print(f"❌ No video files found in '{VIDEOS_PATH}'")
        print(f"   Supported formats: .mp4, .avi, .mov, .mkv")
        return
    
    print(f"\n✓ Found {len(video_files)} videos to analyze")
    
    # Process each video
    results = []
    
    for i, video_file in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_file}")
        
        video_path = os.path.join(VIDEOS_PATH, video_file)
        
        # Initialize result dictionary
        result = {
            'video_filename': video_file,
            'video_path': video_path,
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Quality assessment
        print("  - Assessing quality...")
        quality = assess_video_quality(video_path)
        result.update({
            'resolution': f"{quality['resolution_width']}x{quality['resolution_height']}",
            'fps': quality['fps'],
            'duration_seconds': quality['duration_seconds'],
            'blur_score': quality['blur_score'],
            'quality_status': quality['overall_quality'],
            'quality_issues': '; '.join(quality['quality_issues']) if quality['quality_issues'] else 'None'
        })
        
        # Classification
        print("  - Running classification...")
        classification = classify_video(video_path, processor, model)
        result.update({
            'prediction': classification['prediction'],
            'accident_probability': classification['accident_probability'],
            'non_accident_probability': classification['non_accident_probability'],
            'confidence': classification['confidence'],
            'is_accident': classification['prediction'] == 'accident',
            'processing_error': classification['processing_error']
        })
        
        # Determine if video passes all checks
        passes_quality = quality['overall_quality'] == 'Good'
        passes_classification = (classification['prediction'] == 'accident' and 
                               classification['confidence'] > 0.5)
        
        result['final_verdict'] = 'ACCEPTED' if (passes_quality and passes_classification) else 'REJECTED'
        
        # Print summary
        print(f"  - Prediction: {classification['prediction']} "
              f"(confidence: {classification['confidence']:.2%})")
        print(f"  - Quality: {quality['overall_quality']} "
              f"(blur score: {quality['blur_score']:.1f})")
        print(f"  - Final verdict: {result['final_verdict']}")
        
        results.append(result)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'video_filename', 'final_verdict', 'prediction', 'confidence',
        'accident_probability', 'non_accident_probability', 'is_accident',
        'quality_status', 'blur_score', 'resolution', 'fps', 
        'duration_seconds', 'quality_issues', 'processing_error',
        'processing_time', 'video_path'
    ]
    
    df = df[column_order]
    
    # Sort by confidence (accidents first, then by confidence descending)
    df = df.sort_values(['is_accident', 'confidence'], ascending=[False, False])
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Total videos analyzed: {len(df)}")
    print(f"Accidents detected: {df['is_accident'].sum()}")
    print(f"Videos accepted: {(df['final_verdict'] == 'ACCEPTED').sum()}")
    print(f"Videos rejected: {(df['final_verdict'] == 'REJECTED').sum()}")
    
    if df['is_accident'].sum() > 0:
        avg_confidence = df[df['is_accident']]['confidence'].mean()
        print(f"Average confidence for accidents: {avg_confidence:.2%}")

if __name__ == "__main__":
    print("=== CAR CRASH VIDEO ANALYSIS ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Videos path: {VIDEOS_PATH}")
    print(f"Output CSV: {OUTPUT_CSV}")
    
    analyze_videos()