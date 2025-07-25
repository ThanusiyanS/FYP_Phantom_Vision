from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from youtube_api import download_youtube_videos
from audio_module.predict import predict_and_quality_batch

app = Flask(__name__)
CORS(app)

DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data/initial_video_data.csv')
FINAL_OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data/final_output.csv')
print(DATA_CSV_PATH)

@app.route('/search', methods=['POST'])
def search_youtube():
    print("Searching for YouTube videos")
    data = request.get_json()
    keywords = data.get('keywords', '')
    try:
        results = download_youtube_videos(keywords)
        # After download and extraction, run prediction and quality assessment
        print('predicting and quality assessing')
        predict_and_quality_batch([item['audio_path'] for item in results if item['audio_path']], DATA_CSV_PATH)
        return jsonify({'success': True, 'results': results})
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
