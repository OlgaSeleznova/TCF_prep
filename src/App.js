import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcriptions, setTranscriptions] = useState([]);
  const [status, setStatus] = useState('Ready');
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setStatus('Connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'transcription') {
        setTranscriptions(prev => [...prev, data.text]);
      } else if (data.type === 'status') {
        setStatus(data.message);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      setStatus('Disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('Connection error');
    };

    return () => {
      ws.close();
    };
  }, []);

  const startRecording = () => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({ type: 'start_recording' }));
      setIsRecording(true);
      setTranscriptions([]);
    }
  };

  const stopRecording = () => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({ type: 'stop_recording' }));
      setIsRecording(false);
    }
  };

  const downloadTranscription = () => {
    const text = transcriptions.join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>French Audio Transcription</h1>
        <div className="status">
          Status: <span className={isConnected ? 'connected' : 'disconnected'}>{status}</span>
        </div>
      </header>

      <main className="main-content">
        <div className="controls">
          <button
            onClick={startRecording}
            disabled={isRecording || !isConnected}
            className="btn btn-start"
          >
            {isRecording ? 'Recording...' : 'Start Recording'}
          </button>

          <button
            onClick={stopRecording}
            disabled={!isRecording || !isConnected}
            className="btn btn-stop"
          >
            Stop Recording
          </button>

          {transcriptions.length > 0 && (
            <button
              onClick={downloadTranscription}
              className="btn btn-download"
            >
              Download Transcription
            </button>
          )}
        </div>

        <div className="transcription-area">
          <h2>Transcription Results:</h2>
          <div className="transcription-text">
            {transcriptions.length === 0 ? (
              <p className="placeholder">Click "Start Recording" and speak in French...</p>
            ) : (
              transcriptions.map((text, index) => (
                <p key={index} className="transcription-line">
                  {text}
                </p>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;