# Common Prediction System

This document describes the common prediction system that orchestrates the sequential flow of three modules: **video_module → audio_module → deepfake_detection**.

## Overview

The common prediction system provides a unified interface to run all prediction modules in the correct sequence, ensuring proper data flow and result aggregation.

## Architecture

```
Frontend API Request
        ↓
    main.py (Flask API)
        ↓
common_prediction.py (Orchestrator)
        ↓
┌─────────────────┬─────────────────┬─────────────────────┐
│  video_module   │  audio_module   │ deepfake_detection  │
│                 │                 │                     │
│ • Accident      │ • Crash         │ • Audio Deepfake    │
│   Detection     │   Detection     │ • Video Deepfake    │
│ • Quality       │ • Quality       │ • Final Score       │
│   Assessment    │   Assessment    │   Calculation       │
└─────────────────┴─────────────────┴─────────────────────┘
        ↓
    Results Aggregation
        ↓
    JSON Response to Frontend
```

## Files Structure

```
backend/
├── common_prediction.py          # Main orchestrator script
├── main.py                       # Flask API with new endpoints
├── test_common_prediction.py     # Test script
├── README_PREDICTION.md          # This documentation
├── video_module/
│   └── run_inference.py          # Video accident detection
├── audio_module/
│   └── predict.py                # Audio crash detection
└── deepfake_detection/
    ├── deepfake_audio.py         # Audio deepfake detection
    ├── deepfake_video.py         # Video deepfake detection
    └── deepfake_score.py         # Final score calculation
```

## API Endpoints

### 1. Complete Prediction Pipeline
**Endpoint:** `POST /predict-complete`

Runs the complete prediction pipeline in sequence:
1. Video Module (Accident Detection)
2. Audio Module (Crash Detection)  
3. Deepfake Detection (Audio + Video + Final Score)

**Response:**
```json
{
  "success": true,
  "pipeline_results": {
    "timestamp": "2024-01-01T12:00:00",
    "directory_status": {...},
    "steps": {
      "video_module": {...},
      "audio_module": {...},
      "deepfake_detection": {...}
    },
    "overall_success": true,
    "summary": {...}
  },
  "latest_results": {
    "video_results": {...},
    "deepfake_results": {...},
    "final_deepfake_results": {...}
  },
  "message": "Complete prediction pipeline executed successfully"
}
```

### 2. Get Results
**Endpoint:** `GET /results`

Retrieves the latest prediction results without running the pipeline.

**Response:**
```json
{
  "success": true,
  "results": {
    "video_results": {
      "total_videos": 10,
      "accidents_detected": 3,
      "data": [...]
    },
    "deepfake_results": {
      "total_resources": 10,
      "audio_deepfakes": 2,
      "video_deepfakes": 1,
      "data": [...]
    },
    "final_deepfake_results": {
      "total_resources": 10,
      "deepfakes_detected": 2,
      "data": [...]
    }
  },
  "message": "Latest results retrieved successfully"
}
```

## Usage Examples

### Running the Complete Pipeline

```python
# From Python
from common_prediction import run_complete_prediction_pipeline

results = run_complete_prediction_pipeline()
print(f"Pipeline success: {results['overall_success']}")
```

### Getting Results Only

```python
# From Python
from common_prediction import get_prediction_results

results = get_prediction_results()
print(f"Video results: {results['video_results']}")
```

### From Frontend (JavaScript)

```javascript
// Run complete pipeline
fetch('/predict-complete', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  }
})
.then(response => response.json())
.then(data => {
  console.log('Pipeline results:', data);
});

// Get latest results
fetch('/results')
.then(response => response.json())
.then(data => {
  console.log('Latest results:', data);
});
```

## Pipeline Flow

### Step 1: Video Module
- **Function:** `analyze_videos()` from `video_module/run_inference.py`
- **Purpose:** Accident detection in videos
- **Output:** `video_module/video_output.csv`
- **Key Metrics:** Accident probability, quality assessment

### Step 2: Audio Module  
- **Function:** `predict_and_quality_return()` from `audio_module/predict.py`
- **Purpose:** Crash detection in audio
- **Output:** Updates `data/initial_video_data.csv`
- **Key Metrics:** Crash probability, audio quality

### Step 3: Deepfake Detection
- **3a. Audio Deepfake:** `process_directory_and_save_csv()` from `deepfake_detection/deepfake_audio.py`
- **3b. Video Deepfake:** `process_videos_and_update_csv()` from `deepfake_detection/deepfake_video.py`
- **3c. Final Score:** `calculate_final_deepfake_score()` from `deepfake_detection/deepfake_score.py`
- **Output:** `deepfake_detection/deepfake_score.csv` and `deepfake_detection/final_deepfake_score.csv`
- **Key Metrics:** Deepfake scores, final classification

## Error Handling

The system includes comprehensive error handling:

- **Directory Checks:** Validates required directories exist
- **File Validation:** Ensures input files are present
- **Module Isolation:** Each module runs independently with error isolation
- **Graceful Degradation:** Pipeline continues even if some modules fail
- **Detailed Logging:** Comprehensive error messages and stack traces

## Testing

Run the test script to verify the system:

```bash
cd backend
python test_common_prediction.py
```

This will test:
- Module imports
- Directory structure validation
- Results retrieval functionality

## Configuration

Key configuration paths in `common_prediction.py`:

```python
DATA_DIR = os.path.join(script_dir, 'data')
VIDEOS_DIR = os.path.join(DATA_DIR, 'retrieved-videos')
AUDIO_DIR = os.path.join(DATA_DIR, 'extracted-audios')
INITIAL_CSV_PATH = os.path.join(DATA_DIR, 'initial_video_data.csv')
DEEPFAKE_CSV_PATH = os.path.join(script_dir, 'deepfake_detection', 'deepfake_score.csv')
FINAL_DEEPFAKE_CSV_PATH = os.path.join(script_dir, 'deepfake_detection', 'final_deepfake_score.csv')
```

## Dependencies

Ensure all required dependencies are installed:

```bash
pip install flask flask-cors pandas numpy torch torchvision opencv-python librosa tensorflow tensorflow-hub transformers pillow scikit-learn
```

## Troubleshooting

### Common Issues

1. **Import Errors:** Ensure all module paths are correctly set in `sys.path`
2. **File Not Found:** Check that required directories and files exist
3. **Model Loading:** Verify model files are present in expected locations
4. **Memory Issues:** Large videos may require increased memory allocation

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export DEBUG_PREDICTION=1
```

## Performance Considerations

- **Parallel Processing:** Modules run sequentially for data consistency
- **Memory Management:** Large files are processed in chunks
- **Caching:** Results are cached in CSV files for reuse
- **Error Recovery:** Failed modules don't prevent subsequent steps

## Future Enhancements

- **Parallel Module Execution:** Where data dependencies allow
- **Real-time Processing:** Stream processing for live video feeds
- **Model Versioning:** Support for multiple model versions
- **Result Visualization:** Built-in result visualization tools 