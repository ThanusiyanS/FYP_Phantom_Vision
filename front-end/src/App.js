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
  const [successMessage, setSuccessMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setRetrievedVideos([]);
    setAccidentVideos([]);
    setPredictionResults([]);
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
    setSuccessMessage('');
    setPredictionResults([]);
    setAccidentVideos([]);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await response.json();
      console.log('Prediction Results:', data.results);
      if (data.success) {
        // Set prediction results (all videos sorted by quality)
        setPredictionResults(data.results || []);
        // Filter accident videos from the results
        const accidentVideos = (data.results || []).filter(video => video.is_accident);
        setAccidentVideos(accidentVideos);
        setSuccessMessage(`Prediction completed! Found ${data.results?.length || 0} videos with ${accidentVideos.length} accidents and ${(data.results || []).filter(video => video.is_deepfake).length} deepfakes.`);
        console.log('Prediction Results:', data.results);
        console.log('Accident Videos:', accidentVideos);
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
            {loading ? (
              <>
                Searching...
              </>
            ) : 'Search'}
          </button>
          <button type="button" style={{ padding: '10px 20px', fontSize: '16px', marginTop: '10px' }} onClick={handlePredict} disabled={loading}>
            { 'Run Prediction'}
          </button>
        </form>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        {successMessage && <p style={{ color: 'green', backgroundColor: 'rgba(0,255,0,0.1)', padding: '10px', borderRadius: '5px' }}>{successMessage}</p>}
        <div style={{ display: 'flex', width: '100%', marginTop: '30px', justifyContent: 'center', gap: '40px' }}>
          {/* 1st column: Retrieved videos */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Retrieved Videos</h2>
            {retrievedVideos.length > 0 ? retrievedVideos.map((row, idx) => (
              <div key={row.video_name} className="prediction-card" style={{ 
                marginBottom: 20,
                padding: 10,
                border: '1px solid #ddd',
                borderRadius: 8,
                backgroundColor: '#EEEEEE'
              }}>
                <a href={row.video_url} target="_blank" rel="noopener noreferrer" style={{color:'black',textDecoration:'none',width: '200px',overflow: 'hidden',textOverflow: 'ellipsis',whiteSpace: 'wrap'}}>
                  <img src={getThumbnail(row.video_url)} alt={row.video_name} style={{ width: 200, borderRadius: 8 }} /><br />
                  {row.video_name}
                </a>
              </div>
            )) : <p>No videos</p>}
          </div>
          {/* 2nd column: Predicted accident videos */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Predicted Accident Videos</h2>
            {accidentVideos.length > 0 ? (
              <div>
                <div className="stats-bar">
                  Total Accidents: {accidentVideos.length} | 
                  Deepfakes: {accidentVideos.filter(r => r.is_deepfake).length}
                </div>
                {accidentVideos.map((res, idx) => (
                  <div key={res.video_id || idx} className="prediction-card" style={{ 
                    marginBottom: 20,
                    padding: 10,
                    border: '1px solid #ddd',
                    borderRadius: 8,
                    backgroundColor: '#EEEEEE'
                  }}>
                    {res.video_url && (
                      <a href={res.video_url} target="_blank" rel="noopener noreferrer">
                        <img src={getThumbnail(res.video_url)} alt={res.video_id} style={{ width: 200, borderRadius: 8 }} /><br />
                        <div style={{ fontWeight: 'bold', marginTop: 5 }}>{res.video_id}</div>
                      </a>
                    )}
                    <div style={{ marginTop: 8 }}>
                      <div style={{ marginBottom: 3, color:'black' }}>
                        Quality Score: <span style={{ fontWeight: 'bold', color:'black' }}>{res.quality_score.toFixed(4)}</span>
                      </div>
                      <div style={{ marginBottom: 3, color:'black' }}>
                        Deepfake Score: <span style={{ fontWeight: 'bold', color:'black' }}>{res.deepfake_score.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className={res.is_deepfake ? 'deepfake-badge' : 'real-badge'} style={{ 
                          padding: '2px 6px', 
                          borderRadius: 4, 
                          fontSize: '12px'
                        }}>
                          {res.is_deepfake ? 'DEEPFAKE' : 'REAL'}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : <p>No accident videos detected</p>}
          </div>
          {/* 3rd column: Prediction Results */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <h2>Prediction Results (Real Videos Only)</h2>
            {predictionResults.length > 0 ? (
              <div>
                <div className="stats-bar">
                  Total: {predictionResults.length} | 
                  Real Videos: {predictionResults.filter(r => !r.is_deepfake).length} | 
                  Accidents: {predictionResults.filter(r => r.is_accident && !r.is_deepfake).length}
                </div>
                {predictionResults.filter(res => !res.is_deepfake).map((res, idx) => (
                  <div key={res.video_id || idx} className="prediction-card" style={{ 
                    marginBottom: 15, 
                    padding: 10, 
                    border: '1px solid #ddd', 
                    borderRadius: 8,
                    backgroundColor: res.is_accident ? '#EEEEEE' : '#f8f9fa'
                  }}>
                    <div style={{ fontWeight: 'bold', marginBottom: 5 , color:'black'}}>Video ID: {res.video_id}</div>
                    {res.video_url && (
                      <div style={{ marginBottom: 5, fontSize: '12px' }}>
                        <a href={res.video_url} target="_blank" rel="noopener noreferrer" style={{ color: '#007bff' }}>
                          View Video
                        </a>
                      </div>
                    )}
                    <div style={{ marginBottom: 3 }}>
                      <span className={res.is_accident ? 'accident-badge' : 'no-accident-badge'} style={{ 
                        padding: '2px 6px', 
                        borderRadius: 4, 
                        fontSize: '12px'
                      }}>
                        {res.is_accident ? 'ACCIDENT' : 'NO ACCIDENT'}
                      </span>
                    </div>
                    <div style={{ marginBottom: 3, color:'black' }}>
                      Quality Score: <span style={{ fontWeight: 'bold', color:'black' }}>{res.quality_score.toFixed(4)}</span>
                    </div>
                    <div style={{ marginBottom: 3, color:'black' }}>
                      Deepfake Score: <span style={{ fontWeight: 'bold', color:'black' }}>{res.deepfake_score.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className={res.is_deepfake ? 'deepfake-badge' : 'real-badge'} style={{ 
                        padding: '2px 6px', 
                        borderRadius: 4, 
                        fontSize: '12px'
                      }}>
                        {res.is_deepfake ? 'DEEPFAKE' : 'REAL'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : <p>No prediction results</p>}
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
