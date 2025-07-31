"""
Test script for the common prediction pipeline
"""

import os
import sys
from common_prediction import check_directories, get_prediction_results

def test_directory_check():
    """Test the directory checking functionality"""
    print("Testing directory check...")
    status = check_directories()
    
    print("\nDirectory Status:")
    for dir_name, info in status.items():
        print(f"  {dir_name}:")
        print(f"    Exists: {info['exists']}")
        print(f"    Path: {info['path']}")
        if 'file_count' in info:
            print(f"    File count: {info['file_count']}")
    
    return status

def test_results_retrieval():
    """Test the results retrieval functionality"""
    print("\nTesting results retrieval...")
    results = get_prediction_results()
    
    print("\nResults Summary:")
    for result_type, data in results.items():
        print(f"  {result_type}:")
        if data is None:
            print("    No data available")
        elif isinstance(data, dict) and 'error' in data:
            print(f"    Error: {data['error']}")
        else:
            for key, value in data.items():
                if key != 'data':  # Skip the actual data for brevity
                    print(f"    {key}: {value}")
    
    return results

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from video_module.run_inference import analyze_videos
        print("✅ video_module.run_inference imported successfully")
    except Exception as e:
        print(f"❌ Failed to import video_module.run_inference: {e}")
    
    try:
        from audio_module.predict import predict_and_quality_return, get_all_audio_paths
        print("✅ audio_module.predict imported successfully")
    except Exception as e:
        print(f"❌ Failed to import audio_module.predict: {e}")
    
    try:
        from deepfake_detection.deepfake_audio import process_directory_and_save_csv
        print("✅ deepfake_detection.deepfake_audio imported successfully")
    except Exception as e:
        print(f"❌ Failed to import deepfake_detection.deepfake_audio: {e}")
    
    try:
        from deepfake_detection.deepfake_video import process_videos_and_update_csv
        print("✅ deepfake_detection.deepfake_video imported successfully")
    except Exception as e:
        print(f"❌ Failed to import deepfake_detection.deepfake_video: {e}")
    
    try:
        from deepfake_detection.deepfake_score import calculate_final_deepfake_score
        print("✅ deepfake_detection.deepfake_score imported successfully")
    except Exception as e:
        print(f"❌ Failed to import deepfake_detection.deepfake_score: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING COMMON PREDICTION SCRIPT")
    print("="*60)
    
    # Test imports
    test_imports()
    
    # Test directory check
    test_directory_check()
    
    # Test results retrieval
    test_results_retrieval()
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETED")
    print("="*60) 