import { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // New State for Sidebar Toggle
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null); 
    setError(null);
  };

  const onUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server response was not ok.");

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error(err);
      setError("Unable to communicate with the analysis server.");
    } finally {
      setLoading(false);
    }
  };

  // Helper to normalize backend data structure
  const getProbs = (res) => {
    if (!res) return { real: 0, fake: 0 };
    if (res.details) return { real: res.details.real_prob, fake: res.details.fake_prob };
    if (res.probabilities) return { real: res.probabilities.real, fake: res.probabilities.fake };
    return { real: 0, fake: 0 };
  };

  const probs = getProbs(result);

  return (
    <div className="layout">
      {/* Sidebar / Navigation - Dynamic Class */}
      <nav className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="brand">
          <div className="logo-icon">🧬</div>
          <span className="brand-text">LymphoAI</span>
        </div>
        
        <div className="nav-menu">
          <div className="nav-item active">Analysis</div>
          <div className="nav-item">History</div>
          <div className="nav-item">Settings</div>
        </div>

        <div className="nav-footer">
          <div className="nav-item">Log Out</div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="top-bar">
          <div className="top-bar-left">
            {/* Toggle Button */}
            <button 
              className="toggle-btn" 
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title="Toggle Sidebar"
            >
              ☰
            </button>
            <h1>New Analysis</h1>
          </div>
          <div className="user-profile">Admin User</div>
        </header>

        <div className="dashboard-grid">
          
          {/* Left Panel: Image Input */}
          <div className="panel upload-panel">
            <div className="panel-header">
              <h2>Histology Input</h2>
              <span className="badge-status">Ready</span>
            </div>
            
            <div className="image-stage">
              {preview ? (
                <img src={preview} alt="Histology Slide" className="slide-image" />
              ) : (
                <div className="drop-zone">
                  <span className="drop-icon">⏏</span>
                  <p>Load Slide Image</p>
                </div>
              )}
            </div>

            <div className="controls">
              <input 
                type="file" 
                id="file-upload" 
                onChange={onFileChange} 
                accept="image/*"
                hidden 
              />
              <label htmlFor="file-upload" className="btn btn-secondary">
                {selectedFile ? "Replace File" : "Select File"}
              </label>
              
              <button 
                className="btn btn-primary" 
                onClick={onUpload} 
                disabled={!selectedFile || loading}
              >
                {loading ? "Processing..." : "Run Analysis"}
              </button>
            </div>
            {error && <div className="error-toast">{error}</div>}
          </div>

          {/* Right Panel: Diagnostic Results */}
          <div className="panel result-panel">
            <div className="panel-header">
              <h2>Diagnostic Report</h2>
            </div>

            {!result ? (
              <div className="empty-state">
                <p>Awaiting analysis data...</p>
              </div>
            ) : (
              <div className="report-content">
                
                {/* Primary Classification */}
                <div className={`classification-banner ${result.prediction.toLowerCase()}`}>
                  <span className="label">Classification</span>
                  <span className="value">
                    {result.prediction === 'Fake' ? 'SYNTHETIC' : 'AUTHENTIC TISSUE'}
                  </span>
                </div>

                {/* Confidence Metric */}
                <div className="metric-box">
                  <div className="metric-header">
                    <span>Model Confidence</span>
                    <span className="metric-value">{result.confidence}%</span>
                  </div>
                  <div className="progress-track">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${result.confidence}%` }}
                    ></div>
                  </div>
                </div>

                {/* Probability Breakdown Table */}
                <div className="data-table">
                  <div className="table-row header">
                    <span>Class</span>
                    <span>Probability</span>
                  </div>
                  <div className="table-row">
                    <span className="dot real"></span> Authentic
                    <span className="mono">{probs.real}%</span>
                  </div>
                  <div className="table-row">
                    <span className="dot fake"></span> Synthetic
                    <span className="mono">{probs.fake}%</span>
                  </div>
                </div>

                <div className="disclaimer">
                  AI-assisted result. Verify with clinical protocols.
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;