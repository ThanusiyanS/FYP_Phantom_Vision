"""
Common Prediction Script
Orchestrates the sequential flow: video_module → audio_module → deepfake_detection
"""

import os
import sys
import pandas as pd
from datetime import datetime
import traceback

# Add module paths to sys.path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, 'video_module'))
sys.path.append(os.path.join(script_dir, 'audio_module'))
sys.path.append(os.path.join(script_dir, 'deepfake_detection'))

# Import modules
from video_module.run_inference import analyze_videos
from audio_module.predict import predict_and_quality_return_with_csv, get_all_audio_paths
from deepfake_detection.deepfake_audio import process_directory_and_save_csv
from deepfake_detection.deepfake_video import process_videos_and_update_csv
from deepfake_detection.deepfake_score import calculate_final_deepfake_score

# Configuration
DATA_DIR = os.path.join(script_dir, 'data')
VIDEOS_DIR = os.path.join(DATA_DIR, 'retrieved-videos')
AUDIO_DIR = os.path.join(DATA_DIR, 'extracted-audios')
INITIAL_CSV_PATH = os.path.join(DATA_DIR, 'initial_video_data.csv')
DEEPFAKE_CSV_PATH = os.path.join(script_dir, 'deepfake_detection', 'deepfake_score.csv')
FINAL_DEEPFAKE_CSV_PATH = os.path.join(script_dir, 'deepfake_detection', 'final_deepfake_score.csv')

def check_directories():
    """
    Check if required directories exist and contain files.
    
    Returns:
        dict: Status of each directory
    """
    status = {
        'videos_dir': {'exists': False, 'file_count': 0, 'path': VIDEOS_DIR},
        'audio_dir': {'exists': False, 'file_count': 0, 'path': AUDIO_DIR},
        'data_dir': {'exists': False, 'path': DATA_DIR}
    }
    
    # Check data directory
    if os.path.exists(DATA_DIR):
        status['data_dir']['exists'] = True
    
    # Check videos directory
    if os.path.exists(VIDEOS_DIR):
        status['videos_dir']['exists'] = True
        video_files = [f for f in os.listdir(VIDEOS_DIR) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        status['videos_dir']['file_count'] = len(video_files)
    
    # Check audio directory
    if os.path.exists(AUDIO_DIR):
        status['audio_dir']['exists'] = True
        audio_files = [f for f in os.listdir(AUDIO_DIR) 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        status['audio_dir']['file_count'] = len(audio_files)
    
    return status

def run_video_module():
    """
    Run video module analysis for accident detection.
    
    Returns:
        dict: Status and results of video analysis
    """
    print("\n" + "="*60)
    print("🎬 STEP 1: VIDEO MODULE - Accident Detection")
    print("="*60)
    
    try:
        # Check if videos exist
        if not os.path.exists(VIDEOS_DIR):
            return {
                'success': False,
                'error': f"Videos directory not found: {VIDEOS_DIR}",
                'step': 'video_module'
            }
        
        video_files = [f for f in os.listdir(VIDEOS_DIR) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            return {
                'success': False,
                'error': f"No video files found in {VIDEOS_DIR}",
                'step': 'video_module'
            }
        
        print(f"Found {len(video_files)} video files to analyze")
        
        # Run video analysis
        analyze_videos()
        
        # Check if output CSV was created
        video_output_path = os.path.join(script_dir, 'video_module', 'video_output.csv')
        if os.path.exists(video_output_path):
            df = pd.read_csv(video_output_path)
            print(f"✅ Video analysis completed. Processed {len(df)} videos")
            return {
                'success': True,
                'message': f"Video analysis completed. Processed {len(df)} videos",
                'file_count': len(df),
                'output_file': video_output_path
            }
        else:
            return {
                'success': False,
                'error': "Video analysis completed but output file not found",
                'step': 'video_module'
            }
            
    except Exception as e:
        error_msg = f"Error in video module: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        return {
            'success': False,
            'error': error_msg,
            'step': 'video_module'
        }

def run_audio_module():
    """
    Run audio module analysis for crash detection.
    
    Returns:
        dict: Status and results of audio analysis
    """
    print("\n" + "="*60)
    print("🎵 STEP 2: AUDIO MODULE - Crash Detection")
    print("="*60)
    
    try:
        # Run audio analysis
        print(f"\n🔄 Starting audio analysis...")
        
        # Use the new function that creates CSV (uses default paths)
        results = predict_and_quality_return_with_csv()
        
        # Detailed results analysis
        print(f"\n📊 Audio Analysis Results:")
        print(f"   Total files processed: {len(results)}")
        
        if results:
            # Analyze results
            crash_detected_count = sum(1 for result in results if result.get('is_accident', False))
            quality_scores = [result.get('quality_score', 0) for result in results if 'quality_score' in result]
            
            print(f"   Crashes detected: {crash_detected_count}")
            print(f"   Average quality score: {sum(quality_scores)/len(quality_scores):.4f}" if quality_scores else "   Quality scores: Not available")
            
            # Show sample results
            print(f"\n📋 Sample Results (first 3):")
            for i, result in enumerate(results[:3]):
                video_id = result.get('video_id', 'Unknown')
                prob = result.get('probability', 0)
                is_acc = result.get('is_accident', False)
                qscore = result.get('quality_score', 0)
                print(f"   {i+1}. {video_id}: Prob={prob:.4f}, Crash={is_acc}, Quality={qscore:.4f}")
        
        print(f"\n✅ Audio analysis completed successfully!")
        print(f"   Processed {len(results)} audio files")
        print(f"   Results saved/updated in: {INITIAL_CSV_PATH}")
        
        return {
            'success': True,
            'message': f"Audio analysis completed. Processed {len(results)} audio files",
            'file_count': len(results),
            'results': results,
            'crash_detected': crash_detected_count if results else 0,
            'avg_quality_score': sum(quality_scores)/len(quality_scores) if quality_scores else 0
        }
        
    except Exception as e:
        error_msg = f"Error in audio module: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error details: {str(e)}")
        print(traceback.format_exc())
        return {
            'success': False,
            'error': error_msg,
            'step': 'audio_module'
        }

def delete_non_accident_videos():
    """
    Delete videos and their corresponding audio files if they are not accidents
    based on video module OR audio module predictions.
    
    Returns:
        dict: Status and results of deletion process
    """
    print("\n" + "="*60)
    print("🗑️  STEP 2.5: DELETE NON-ACCIDENT VIDEOS")
    print("="*60)
    
    try:
        # Load video and audio prediction results
        video_output_path = os.path.join(script_dir, 'video_module', 'video_output.csv')
        audio_output_path = os.path.join(script_dir, 'audio_module', 'audio_prediction_results.csv')
        
        if not os.path.exists(video_output_path):
            return {
                'success': False,
                'error': 'Video results file not found'
            }
        
        if not os.path.exists(audio_output_path):
            return {
                'success': False,
                'error': 'Audio results file not found'
            }
        
        video_df = pd.read_csv(video_output_path)
        audio_df = pd.read_csv(audio_output_path)
        
        deleted_count = 0
        errors = []
        
        # Process each video
        for _, video_row in video_df.iterrows():
            video_filename = video_row['video_filename']
            video_id = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
            
            # Get video accident status
            video_is_accident = video_row['is_accident']
            
            # Get audio accident status
            audio_row = audio_df[audio_df['video_id'] == video_id]
            audio_is_accident = audio_row.iloc[0]['is_accident'] if not audio_row.empty else False
            
            # Check if video is NOT an accident (using OR logic)
            is_not_accident = not (video_is_accident or audio_is_accident)
            
            if is_not_accident:
                # Delete video file
                video_path = os.path.join(script_dir, '.', 'data', 'retrieved-videos', video_filename)
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        print(f"  - Deleted non-accident video: {video_filename}")
                    else:
                        print(f"  - Warning: Video file not found: {video_path}")
                except Exception as e:
                    error_msg = f"Could not delete video {video_filename}: {str(e)}"
                    errors.append(error_msg)
                    print(f"  - Error: {error_msg}")
                
                # Delete corresponding audio file
                audio_filename = video_id + '.mp3'
                audio_path = os.path.join(script_dir, '..', 'data', 'extracted-audios', audio_filename)
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        print(f"  - Deleted corresponding audio: {audio_filename}")
                    else:
                        print(f"  - Warning: Audio file not found: {audio_path}")
                except Exception as e:
                    error_msg = f"Could not delete audio {audio_filename}: {str(e)}"
                    errors.append(error_msg)
                    print(f"  - Error: {error_msg}")
                
                deleted_count += 1
        
        print(f"\n✅ Deletion process completed:")
        print(f"   - Videos deleted: {deleted_count}")
        print(f"   - Errors encountered: {len(errors)}")
        
        return {
            'success': True,
            'deleted_count': deleted_count,
            'errors': errors,
            'message': f"Successfully deleted {deleted_count} non-accident videos"
        }
        
    except Exception as e:
        error_msg = f"Error in deletion process: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }

def run_deepfake_detection():
    """
    Run deepfake detection modules (audio, video, and final score calculation).
    
    Returns:
        dict: Status and results of deepfake detection
    """
    print("\n" + "="*60)
    print("🔍 STEP 3: DEEPFAKE DETECTION")
    print("="*60)
    
    results = {
        'audio_deepfake': {'success': False},
        'video_deepfake': {'success': False},
        'final_score': {'success': False}
    }
    
    try:
        # Step 3a: Audio Deepfake Detection
        print("\n--- 3a: Audio Deepfake Detection ---")
        try:
            process_directory_and_save_csv(AUDIO_DIR)
            results['audio_deepfake'] = {
                'success': True,
                'message': "Audio deepfake detection completed"
            }
            print("✅ Audio deepfake detection completed")
        except Exception as e:
            error_msg = f"Error in audio deepfake detection: {str(e)}"
            print(f"❌ {error_msg}")
            results['audio_deepfake'] = {
                'success': False,
                'error': error_msg
            }
        
        # Step 3b: Video Deepfake Detection
        print("\n--- 3b: Video Deepfake Detection ---")
        try:
            success = process_videos_and_update_csv()
            if success:
                results['video_deepfake'] = {
                    'success': True,
                    'message': "Video deepfake detection completed"
                }
                print("✅ Video deepfake detection completed")
            else:
                results['video_deepfake'] = {
                    'success': False,
                    'error': "Video deepfake detection failed"
                }
                print("❌ Video deepfake detection failed")
        except Exception as e:
            error_msg = f"Error in video deepfake detection: {str(e)}"
            print(f"❌ {error_msg}")
            results['video_deepfake'] = {
                'success': False,
                'error': error_msg
            }
        
        # Step 3c: Final Deepfake Score Calculation
        print("\n--- 3c: Final Deepfake Score Calculation ---")
        try:
            calculate_final_deepfake_score()
            results['final_score'] = {
                'success': True,
                'message': "Final deepfake score calculation completed"
            }
            print("✅ Final deepfake score calculation completed")
        except Exception as e:
            error_msg = f"Error in final score calculation: {str(e)}"
            print(f"❌ {error_msg}")
            results['final_score'] = {
                'success': False,
                'error': error_msg
            }
        
        # Check overall success
        overall_success = all(result['success'] for result in results.values())
        
        return {
            'success': overall_success,
            'results': results,
            'message': "Deepfake detection completed" if overall_success else "Deepfake detection completed with errors"
        }
        
    except Exception as e:
        error_msg = f"Error in deepfake detection: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        return {
            'success': False,
            'error': error_msg,
            'step': 'deepfake_detection'
        }

def run_complete_prediction_pipeline():
    """
    Run the complete prediction pipeline: video → audio → deepfake detection.
    
    Returns:
        dict: Complete results from all modules
    """
    print("\n" + "="*80)
    print("🚀 STARTING COMPLETE PREDICTION PIPELINE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check directory status
    print("\n📁 Checking directory status...")
    dir_status = check_directories()
    for dir_name, status in dir_status.items():
        if status['exists']:
            print(f"✅ {dir_name}: {status.get('file_count', 'N/A')} files")
        else:
            print(f"❌ {dir_name}: Not found")
    
    # Initialize results
    pipeline_results = {
        'timestamp': datetime.now().isoformat(),
        'directory_status': dir_status,
        'steps': {},
        'overall_success': False,
        'summary': {}
    }
    
    # Step 1: Video Module
    video_result = run_video_module()
    pipeline_results['steps']['video_module'] = video_result
    
    if not video_result['success']:
        pipeline_results['summary']['error'] = f"Pipeline failed at video module: {video_result['error']}"
        return pipeline_results
    
    # Step 2: Audio Module
    audio_result = run_audio_module()
    pipeline_results['steps']['audio_module'] = audio_result
    
    if not audio_result['success']:
        pipeline_results['summary']['error'] = f"Pipeline failed at audio module: {audio_result['error']}"
        return pipeline_results
    
    # Step 2.5: Delete Non-Accident Videos
    deletion_result = delete_non_accident_videos()
    pipeline_results['steps']['deletion'] = deletion_result
    
    if not deletion_result['success']:
        pipeline_results['summary']['warning'] = f"Deletion step had issues: {deletion_result.get('error', 'Unknown error')}"
    
    # Step 3: Deepfake Detection
    deepfake_result = run_deepfake_detection()
    pipeline_results['steps']['deepfake_detection'] = deepfake_result
    
    if not deepfake_result['success']:
        pipeline_results['summary']['warning'] = f"Pipeline completed with deepfake detection errors: {deepfake_result.get('error', 'Unknown error')}"
    
    # Final summary
    print("\n" + "="*80)
    print("📊 PIPELINE SUMMARY")
    print("="*80)
    
    success_count = sum(1 for step in pipeline_results['steps'].values() if step['success'])
    total_steps = len(pipeline_results['steps'])
    
    pipeline_results['overall_success'] = success_count == total_steps
    pipeline_results['summary']['success_rate'] = f"{success_count}/{total_steps} steps successful"
    
    print(f"Overall Success: {'✅ YES' if pipeline_results['overall_success'] else '❌ NO'}")
    print(f"Success Rate: {pipeline_results['summary']['success_rate']}")
    
    # Check output files
    output_files = []
    if os.path.exists(os.path.join(script_dir, 'video_module', 'video_output.csv')):
        output_files.append('video_output.csv')
    if os.path.exists(DEEPFAKE_CSV_PATH):
        output_files.append('deepfake_score.csv')
    if os.path.exists(FINAL_DEEPFAKE_CSV_PATH):
        output_files.append('final_deepfake_score.csv')
    
    pipeline_results['summary']['output_files'] = output_files
    print(f"Output Files: {', '.join(output_files) if output_files else 'None'}")
    
    print("\n" + "="*80)
    print("🏁 PIPELINE COMPLETED")
    print("="*80)
    
    return pipeline_results

def get_prediction_results():
    """
    Get the latest prediction results from output files.
    
    Returns:
        dict: Latest results from all modules
    """
    results = {
        'video_results': None,
        'deepfake_results': None,
        'final_deepfake_results': None
    }
    
    # Get video results
    video_output_path = os.path.join(script_dir, 'video_module', 'video_output.csv')
    if os.path.exists(video_output_path):
        try:
            df = pd.read_csv(video_output_path)
            results['video_results'] = {
                'total_videos': len(df),
                'accidents_detected': len(df[df['is_accident'] == True]) if 'is_accident' in df.columns else 0,
                'data': df.to_dict('records')
            }
        except Exception as e:
            results['video_results'] = {'error': str(e)}
    
    # Get deepfake results
    if os.path.exists(DEEPFAKE_CSV_PATH):
        try:
            df = pd.read_csv(DEEPFAKE_CSV_PATH)
            results['deepfake_results'] = {
                'total_resources': len(df),
                'audio_deepfakes': len(df[df['is_audio_deepfake'] == 1]) if 'is_audio_deepfake' in df.columns else 0,
                'video_deepfakes': len(df[df['is_video_deepfake'] == 1]) if 'is_video_deepfake' in df.columns else 0,
                'data': df.to_dict('records')
            }
        except Exception as e:
            results['deepfake_results'] = {'error': str(e)}
    
    # Get final deepfake results
    if os.path.exists(FINAL_DEEPFAKE_CSV_PATH):
        try:
            df = pd.read_csv(FINAL_DEEPFAKE_CSV_PATH)
            results['final_deepfake_results'] = {
                'total_resources': len(df),
                'deepfakes_detected': len(df[df['deepfake_label'] == 1]) if 'deepfake_label' in df.columns else 0,
                'data': df.to_dict('records')
            }
        except Exception as e:
            results['final_deepfake_results'] = {'error': str(e)}
    
    return results

def get_integrated_results():
    """
    Integrate all three CSV files and return combined results for frontend.
    
    Returns:
        dict: Integrated results with video_id, is_accident, quality_score, deepfake_score, deepfake_label
    """
    try:
        # File paths
        video_output_path = os.path.join(script_dir, 'video_module', 'video_output.csv')
        audio_output_path = os.path.join(script_dir, 'audio_module', 'audio_prediction_results.csv')
        deepfake_output_path = os.path.join(script_dir, 'deepfake_detection', 'final_deepfake_score.csv')
        
        # Check if all files exist
        if not os.path.exists(video_output_path):
            return {'error': 'Video results file not found'}
        if not os.path.exists(audio_output_path):
            return {'error': 'Audio results file not found'}
        if not os.path.exists(deepfake_output_path):
            return {'error': 'Deepfake results file not found'}
        
        # Load all CSV files
        video_df = pd.read_csv(video_output_path)
        audio_df = pd.read_csv(audio_output_path)
        deepfake_df = pd.read_csv(deepfake_output_path)
        
        # Create integrated results
        integrated_results = []
        
        # Process each video
        for _, video_row in video_df.iterrows():
            video_filename = video_row['video_filename']
            video_id = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
            
            # Get video data
            video_is_accident = video_row['is_accident']
            blur_score = video_row.get('blur_score', 0)  # Get blur score
            
            # Scale blur score to 0-1 and deduct from 1 to get quality score
            # Higher blur score = lower quality, so we invert it
            if blur_score > 0:
                # Normalize to 0-1 range (assuming blur_score can be any positive value)
                normalized_blur = min(blur_score / 100.0, 1.0)  # Scale by 100 as reasonable max
                video_quality = 1.0 - normalized_blur
            else:
                video_quality = 1.0  # Perfect quality if no blur detected
            
            # Find corresponding audio data
            audio_row = audio_df[audio_df['video_id'] == video_id]
            if not audio_row.empty:
                audio_is_accident = audio_row.iloc[0]['is_accident']
                audio_quality = audio_row.iloc[0]['quality_score']
                video_url = audio_row.iloc[0].get('video_url', '')
            else:
                audio_is_accident = False
                audio_quality = 0.0
                video_url = ''
            
            # Find corresponding deepfake data
            deepfake_row = deepfake_df[deepfake_df['video_file'] == video_filename]
            if not deepfake_row.empty:
                deepfake_score = deepfake_row.iloc[0]['final_deepfake_score']
                deepfake_label = deepfake_row.iloc[0]['deepfake_label']
            else:
                deepfake_score = 0.0
                deepfake_label = 0
            
            # Calculate integrated values
            is_accident = video_is_accident or audio_is_accident  # OR condition
            
            # Calculate deepfake quality component (deduct from 1 to invert the relationship)
            # Higher deepfake score = lower quality, so we invert it
            deepfake_quality = 1.0 - deepfake_score
            
            # Calculate weighted average quality score
            # Weights: video_quality (0.5), audio_quality (0.3), deepfake_quality (0.2)
            quality_score = (0.4 * video_quality) + (0.3 * audio_quality) + (0.3 * deepfake_quality)
            
            # Create result object
            result = {
                'video_id': video_id,
                'video_url': video_url,
                'is_accident': bool(is_accident),
                'quality_score': round(quality_score, 4),
                'deepfake_score': round(deepfake_score, 4),
                'deepfake_label': int(deepfake_label),
            }
            
            integrated_results.append(result)
        
        # Sort by video_id for consistency
        integrated_results.sort(key=lambda x: x['video_id'])
        
        return {
            'success': True,
            'total_videos': len(integrated_results),
            'accidents_detected': sum(1 for r in integrated_results if r['is_accident']),
            'deepfakes_detected': sum(1 for r in integrated_results if r['deepfake_label'] == 1),
            'results': integrated_results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error integrating results: {str(e)}',
            'traceback': traceback.format_exc()
        }

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_prediction_pipeline()
    print(f"\nPipeline completed with overall success: {results['overall_success']}") 