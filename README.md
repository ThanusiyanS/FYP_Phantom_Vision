# Final System v1

## Project Overview
This project is a full-stack system for searching, downloading, and analyzing YouTube videos for accident detection. It consists of a Python backend (Flask) and a React frontend.

---

## Backend Setup (Flask + ML)

### 1. Prerequisites
- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (for robust YouTube downloads)
- ffmpeg (for audio extraction)

#### Install yt-dlp and ffmpeg (macOS example):
```sh
brew install yt-dlp ffmpeg
```
Or with pip (yt-dlp only):
```sh
pip install yt-dlp
```

### 2. Install Python Dependencies
From the project root:
```sh
pip install -r requirements.txt
```

### 3. Model Files
- Place your trained model file (`crash_classifier_v1.keras`) in the `backend/` directory.

### 4. Run the Backend
From the project root:
```sh
python3 backend/main.py
```
- The backend will start on `http://127.0.0.1:5000` by default.

---

## Frontend Setup (React)

### 1. Prerequisites
- Node.js (v16+ recommended)
- npm (comes with Node.js)

### 2. Install Frontend Dependencies
From the project root:
```sh
cd front-end
npm install
```

### 3. Run the Frontend
```sh
npm start
```
- The frontend will start on `http://localhost:3000`.

---

## Usage
1. Open the frontend in your browser: [http://localhost:3000](http://localhost:3000)
2. Enter a search term and submit.
3. The system will:
   - Search YouTube, download videos, extract audio, and run accident prediction.
   - Display results in three columns: retrieved videos, predicted accident videos, and CSV data.

---

## Notes
- Every new search clears previous downloads and CSV data.
- If you encounter errors, check the backend terminal for Python errors and the browser console for frontend errors.
- Make sure all dependencies (yt-dlp, ffmpeg, moviepy, etc.) are installed and available in your PATH.

---

## Troubleshooting
- **500 Internal Server Error:** Check backend logs for Python errors.
- **CORS errors:** Ensure Flask-CORS is enabled in the backend.
- **yt-dlp or ffmpeg not found:** Make sure they are installed and accessible from your shell.
- **Model file not found:** Ensure `crash_classifier_v1.keras` is in the `backend/` directory.

---

## License
This project is for educational and research purposes. 