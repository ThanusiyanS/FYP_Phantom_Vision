from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from youtube_api import download_youtube_videos
from audio_module.predict import predict_and_quality_return, get_all_audio_paths

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
def predict_api():
    audio_dir = os.path.join(os.path.dirname(__file__), 'data/extracted-audios')
    csv_path = os.path.join(os.path.dirname(__file__), 'data/initial_video_data.csv')
    audio_paths = get_all_audio_paths(audio_dir)
    print(f"[PREDICT] Found audio files: {audio_paths}")
    try:
        results = predict_and_quality_return(audio_paths, csv_path)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
