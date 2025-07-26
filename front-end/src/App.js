import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [retrievedVideos, setRetrievedVideos] = useState([]);
  const [accidentVideos, setAccidentVideos] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [error, setError] = useState('');
  const [predictionResults, setPredictionResults] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setRetrievedVideos([]);
    setAccidentVideos([]);
    setCsvData([]);
    try {
      // 1. Search and download videos
      const response = await fetch('http://127.0.0.1:5000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keywords: query })
      });
      if (!response.ok) {
        const text = await response.text();
        setError(`Backend error: ${response.status}`);
        console.error('Non-OK response:', response.status, text);
        setLoading(false);
        return;
      }
      const data = await response.json();
      console.log('Backend response:', data);
      if (data.success) {
        const videos = data.videos;
        setRetrievedVideos(videos);
        console.log('retrievedVideos', retrievedVideos)
        console.log('videos', videos)
      } else {
        setError(data.error || 'Unknown error occurred');
        console.error('Backend error:', data.error);
      }
      // // 2. Predict accident videos
      // const predictRes = await fetch('http://127.0.0.1:5000/predict', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ video_links: data.results.map(item => item.audio_path) })
      // });
      // const predictData = await predictRes.json();
      // if (predictData.success) {
      //   setAccidentVideos(predictData.accident_videos || []);
      // }
      // // 3. Fetch CSV preview
      // const csvRes = await fetch('http://127.0.0.1:5000/csv-preview');
      // const csvJson = await csvRes.json();
      // if (csvJson.success) {
      //   setCsvData(csvJson.rows || []);
      // }
    } catch (err) {
      setError('Failed to connect to backend');
      console.error('Fetch error:', err);
    }
    setLoading(false);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setPredictionResults([]);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}) // No audio_paths sent
      });
      const data = await response.json();
      if (data.success) {
        setPredictionResults(data.results);
      } else {
        setError(data.error || 'Prediction error');
      }
    } catch (err) {
      setError('Failed to connect to backend for prediction');
      console.error('Prediction fetch error:', err);
    }
    setLoading(false);
  };

  // Helper to get YouTube thumbnail from URL
  const getThumbnail = (url) => {
    const match = url.match(/v=([\w-]+)/);
    if (match) {
      return `https://img.youtube.com/vi/${match[1]}/0.jpg`;
    }
    return '';
  };

  // Sort prediction results: accidents first (by quality_score desc), then non-accidents (by quality_score desc)
  const sortedPredictionResults = [...predictionResults].sort((a, b) => {
    if (b.is_accident !== a.is_accident) return b.is_accident - a.is_accident;
    return (b.quality_score || 0) - (a.quality_score || 0);
  });

  return (
    <div className="App">
      <header className="App-header">
        <h1>YouTube Video Search</h1>
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Enter search term..."
            style={{ padding: '10px', fontSize: '16px', width: '300px', marginBottom: '10px' }}
          />
          <button type="submit" style={{ padding: '10px 20px', fontSize: '16px' }} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
          <button type="button" style={{ padding: '10px 20px', fontSize: '16px', marginTop: '10px' }} onClick={handlePredict} disabled={false}>
            {loading ? 'Running Prediction...' : 'Run Prediction'}
          </button>
        </form>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <div style={{ display: 'flex', width: '100%', marginTop: '30px', justifyContent: 'center', gap: '40px' }}>
          {/* 1st column: Retrieved videos */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Retrieved Videos</h2>
            {retrievedVideos.length > 0 ? retrievedVideos.map((row, idx) => (
              <div key={row.video_name} style={{ marginBottom: 20 }}>
                <a href={row.video_url} target="_blank" rel="noopener noreferrer">
                  <img src={getThumbnail(row.video_url)} alt={row.video_name} style={{ width: 200, borderRadius: 8 }} /><br />
                  {row.video_name}
                </a>
              </div>
            )) : <p>No videos</p>}
          </div>
          {/* 2nd column: Predicted accident videos */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Predicted Accident Videos</h2>
            {sortedPredictionResults.filter(res => res.is_accident).length > 0 ? 
              sortedPredictionResults
                .filter(res => res.is_accident)
                .map((res, idx) => (
                  <div key={res.video_id || idx} style={{ marginBottom: 20 }}>
                    {res.video_url && (
                      <a href={res.video_url} target="_blank" rel="noopener noreferrer">
                        <img src={getThumbnail(res.video_url)} alt={res.video_name || res.video_id} style={{ width: 200, borderRadius: 8 }} /><br />
                        {res.video_name || res.video_id}
                      </a>
                    )}
                    <div>Probability: {res.probability !== undefined ? res.probability.toFixed(4) : 'N/A'}</div>
                    <div>Quality Score: {res.quality_score !== undefined ? res.quality_score.toFixed(4) : 'N/A'}</div>
                  </div>
                )) : <p>No accident videos detected</p>}
          </div>
          {/* 3rd column: CSV Data */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Prediction Results</h2>
            {sortedPredictionResults.length > 0 ? sortedPredictionResults.map((res, idx) => (
              <div key={res.video_id || idx} style={{ marginBottom: 20 }}>
                <div>Video ID: {res.video_id}</div>
                {res.video_name && res.video_url && (
                  <div>
                    <a href={res.video_url} target="_blank" rel="noopener noreferrer">
                      {res.video_name}
                    </a>
                  </div>
                )}
                <div>Probability: {res.probability !== undefined ? res.probability.toFixed(4) : 'N/A'}</div>
                <div>Is Accident: {res.is_accident ? 'Yes' : 'No'}</div>
                <div>Quality Score: {res.quality_score !== undefined ? res.quality_score.toFixed(4) : 'N/A'}</div>
                {res.error && <div style={{ color: 'red' }}>Error: {res.error}</div>}
              </div>
            )) : <p>No prediction results</p>}
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
