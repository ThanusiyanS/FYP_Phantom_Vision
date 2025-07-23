import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [retrievedVideos, setRetrievedVideos] = useState([]);
  const [accidentVideos, setAccidentVideos] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [error, setError] = useState('');

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
        const videos = data.results.map(item => {
          const filename = item.video_path.split('/').pop();
          return {
            video_path: item.video_path,
            audio_path: item.audio_path,
            filename,
          };
        });
        setRetrievedVideos(videos);
      } else {
        setError(data.error || 'Unknown error occurred');
        console.error('Backend error:', data.error);
      }
      // 2. Predict accident videos
      const predictRes = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_links: data.results.map(item => item.audio_path) })
      });
      const predictData = await predictRes.json();
      if (predictData.success) {
        setAccidentVideos(predictData.accident_videos || []);
      }
      // 3. Fetch CSV preview
      const csvRes = await fetch('http://127.0.0.1:5000/csv-preview');
      const csvJson = await csvRes.json();
      if (csvJson.success) {
        setCsvData(csvJson.rows || []);
      }
    } catch (err) {
      setError('Failed to connect to backend');
      console.error('Fetch error:', err);
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
        </form>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <div style={{ display: 'flex', width: '100%', marginTop: '30px', justifyContent: 'center', gap: '40px' }}>
          {/* 1st column: Retrieved videos */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Retrieved Videos</h2>
            {csvData.length > 0 ? csvData.map((row, idx) => (
              <div key={row.video_id} style={{ marginBottom: 20 }}>
                <a href={row.video_url} target="_blank" rel="noopener noreferrer">
                  <img src={getThumbnail(row.video_url)} alt={row.video_id} style={{ width: 200, borderRadius: 8 }} /><br />
                  {row.video_id}
                </a>
              </div>
            )) : <p>No videos</p>}
          </div>
          {/* 2nd column: Predicted accident videos */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Predicted Accident Videos</h2>
            {accidentVideos.length > 0 ? accidentVideos.map((row, idx) => (
              <div key={row.video_path || idx} style={{ marginBottom: 20 }}>
                <a href={row.video_url || '#'} target="_blank" rel="noopener noreferrer">
                  <img src={getThumbnail(row.video_url || '')} alt={row.video_id || ''} style={{ width: 200, borderRadius: 8 }} /><br />
                  {row.video_id || row.video_path}
                </a>
                <div>Probability: {row.probability !== undefined ? row.probability : 'N/A'}</div>
              </div>
            )) : <p>No accident videos</p>}
          </div>
          {/* 3rd column: CSV Data */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>CSV Data</h2>
            {csvData.length > 0 ? csvData.map((row, idx) => (
              <div key={row.video_id} style={{ marginBottom: 20 }}>
                <a href={row.video_url} target="_blank" rel="noopener noreferrer">
                  {row.video_id}
                </a>
                <div>Probability: {row.probability !== undefined ? row.probability : 'N/A'}</div>
              </div>
            )) : <p>No CSV data</p>}
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
