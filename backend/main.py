from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from youtube_api import download_youtube_videos  # You may need to adjust import based on actual function name
from predict import predict_audio_batch

app = Flask(__name__)
CORS(app)

DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data/video_data.csv')
print(DATA_CSV_PATH)

@app.route('/search', methods=['POST'])
def search_youtube():
    print("Searching for YouTube videos")
    data = request.get_json()
    keywords = data.get('keywords', '')
    # Call your YouTube API script
    try:
        results = download_youtube_videos(keywords)  # Should return list of dicts with 'title', 'url', 'thumbnail_url'
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    video_links = data.get('video_links', [])
    try:
        predictions = predict_audio_batch(video_links)
        accident_videos = [v for v in predictions if v.get('is_accident')]
        return jsonify({'success': True, 'accident_videos': accident_videos})
    except Exception as e:
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
