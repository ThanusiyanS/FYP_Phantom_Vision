from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from youtube_api import download_youtube_videos
from audio_module.predict import predict_and_quality_return, get_all_audio_paths
from common_prediction import run_complete_prediction_pipeline, get_prediction_results, get_integrated_results

app = Flask(__name__)
CORS(app)

DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data/initial_video_data.csv')
FINAL_OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data/final_output.csv')
print(DATA_CSV_PATH)

@app.route('/search', methods=['POST'])
def search_youtube():
    print(f"[{__import__('datetime').datetime.now()}] New search request received")
    data = request.get_json()
    keywords = data.get('keywords', '')
    print(f"Searching for: {keywords}")
    try:
        download_result = download_youtube_videos(keywords)
        print(f"Download completed. Found {len(download_result['urls'])} videos")
        # After download and extraction, run prediction and quality assessment
        print('predicting and quality assessing', [item['audio_path'] for item in download_result['results'] if item['audio_path']])
        print('DATA_CSV_PATH', DATA_CSV_PATH)
        # predict_and_quality_batch([item['audio_path'] for item in download_result['results'] if item['audio_path']], DATA_CSV_PATH)
        # Create videos array with video_url and video_name
        videos = []
        for i in range(len(download_result['urls'])):
            videos.append({
                'video_url': download_result['urls'][i],
                'video_name': download_result['video_names'][i]
            })
        
        response = {
            'success': True, 
            'videos': videos
        }
        print(f"Returning response with {len(videos)} videos")
        return jsonify(response)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/csv-preview', methods=['GET'])
def csv_preview():
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        preview = df.head(10).to_dict(orient='records')
        columns = list(df.columns)
        return jsonify({'success': True, 'columns': columns, 'rows': preview})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict_complete_api():
    """
    Run the complete prediction pipeline and return formatted results for frontend
    """
    print(f"[{__import__('datetime').datetime.now()}] Complete prediction pipeline requested")
    
    try:
        # Run the complete prediction pipeline
        pipeline_results = run_complete_prediction_pipeline()
        
        if not pipeline_results['overall_success']:
            return jsonify({
                'success': False,
                'error': pipeline_results.get('error', 'Pipeline failed'),
                'message': 'Prediction pipeline failed'
            }), 500
        
        # Get integrated results
        integrated_results = get_integrated_results()
        
        if 'error' in integrated_results:
            return jsonify({
                'success': False,
                'error': integrated_results['error'],
                'message': 'Failed to integrate results'
            }), 500
        
        # Sort all results by quality score (highest first)
        sorted_results = sorted(
            integrated_results['results'], 
            key=lambda x: x['quality_score'], 
            reverse=True
        )
        
        # Format results for frontend - Only the specific fields requested
        formatted_results = []
        for result in sorted_results:
            formatted_results.append({
                'video_id': result['video_id'],
                'video_url': result['video_url'],
                'is_accident': result['is_accident'],
                'quality_score': result['quality_score'],
                'deepfake_score': result['deepfake_score'],
                'is_deepfake': bool(result['deepfake_label'])
            })
        
        response = {
            'success': True,
            'message': 'Prediction pipeline completed successfully',
            'results': formatted_results
        }
        
        print(f"[PREDICT] Pipeline completed successfully with {len(formatted_results)} videos")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in complete prediction pipeline: {str(e)}"
        print(f"[PREDICT] Error: {error_msg}")
        return jsonify({
            'success': False, 
            'error': error_msg,
            'message': 'Failed to execute complete prediction pipeline'
        }), 500

@app.route('/results', methods=['GET'])
def get_results_api():
    """
    Get the latest prediction results without running the pipeline
    """
    try:
        results = get_prediction_results()
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Latest results retrieved successfully'
        })
    except Exception as e:
        error_msg = f"Error retrieving results: {str(e)}"
        print(f"[RESULTS] Error: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'message': 'Failed to retrieve results'
        }), 500

@app.route('/integrated-results', methods=['GET'])
def get_integrated_results_api():
    """
    Get integrated results combining video, audio, and deepfake detection
    """
    try:
        integrated_results = get_integrated_results()
        
        if 'error' in integrated_results:
            return jsonify({
                'success': False,
                'error': integrated_results['error'],
                'message': 'Failed to integrate results'
            }), 500
        
        return jsonify({
            'success': True,
            'data': integrated_results,
            'message': 'Integrated results retrieved successfully'
        })
        
    except Exception as e:
        error_msg = f"Error retrieving integrated results: {str(e)}"
        print(f"[INTEGRATED-RESULTS] Error: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'message': 'Failed to retrieve integrated results'
        }), 500

@app.route('/prediction-results', methods=['GET'])
def get_prediction_results_api():
    """
    Get prediction results sorted by quality score (highest first)
    """
    try:
        integrated_results = get_integrated_results()
        
        if 'error' in integrated_results:
            return jsonify({
                'success': False,
                'error': integrated_results['error'],
                'message': 'Failed to get prediction results'
            }), 500
        
        # Sort by quality score (highest first)
        sorted_results = sorted(
            integrated_results['results'], 
            key=lambda x: x['quality_score'], 
            reverse=True
        )
        
        # Format results for frontend
        formatted_results = []
        for result in sorted_results:
            formatted_results.append({
                'video_id': result['video_id'],
                'is_accident': result['is_accident'],
                'quality_score': result['quality_score'],
                'deepfake_score': result['deepfake_score'],
                'is_deepfake': bool(result['deepfake_label'])
            })
        
        return jsonify({
            'success': True,
            'data': {
                'total_videos': len(formatted_results),
                'accidents_detected': sum(1 for r in formatted_results if r['is_accident']),
                'deepfakes_detected': sum(1 for r in formatted_results if r['is_deepfake']),
                'results': formatted_results
            },
            'message': 'Prediction results retrieved successfully'
        })
        
    except Exception as e:
        error_msg = f"Error retrieving prediction results: {str(e)}"
        print(f"[PREDICTION-RESULTS] Error: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'message': 'Failed to retrieve prediction results'
        }), 500

@app.route('/accident-videos', methods=['GET'])
def get_accident_videos_api():
    """
    Get only accident videos sorted by quality score (highest first)
    """
    try:
        integrated_results = get_integrated_results()
        
        if 'error' in integrated_results:
            return jsonify({
                'success': False,
                'error': integrated_results['error'],
                'message': 'Failed to get accident videos'
            }), 500
        
        # Filter only accident videos and sort by quality score (highest first)
        accident_videos = [
            result for result in integrated_results['results'] 
            if result['is_accident']
        ]
        
        sorted_accident_videos = sorted(
            accident_videos, 
            key=lambda x: x['quality_score'], 
            reverse=True
        )
        
        # Format results for frontend
        formatted_results = []
        for result in sorted_accident_videos:
            formatted_results.append({
                'video_id': result['video_id'],
                'video_url': result['video_url'],
                'quality_score': result['quality_score'],
                'deepfake_score': result['deepfake_score'],
                'is_deepfake': bool(result['deepfake_label'])
            })
        
        return jsonify({
            'success': True,
            'data': {
                'total_accident_videos': len(formatted_results),
                'deepfakes_in_accidents': sum(1 for r in formatted_results if r['is_deepfake']),
                'results': formatted_results
            },
            'message': 'Accident videos retrieved successfully'
        })
        
    except Exception as e:
        error_msg = f"Error retrieving accident videos: {str(e)}"
        print(f"[ACCIDENT-VIDEOS] Error: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'message': 'Failed to retrieve accident videos'
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
