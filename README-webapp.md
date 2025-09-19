# French Transcription Web App

A web application for real-time French audio transcription using Whisper AI.

## Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements-web.txt
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Build Frontend
```bash
npm run build
```

### 4. Start the Application
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### 5. Open Browser
Navigate to: `http://localhost:8000`

## Usage

1. **Start Recording**: Click the green "Start Recording" button
2. **Speak in French**: The app will transcribe your speech in real-time
3. **Stop Recording**: Click the red "Stop Recording" button
4. **Download**: Use the blue "Download Transcription" button to save results

## Features

- Real-time French audio transcription
- Clean, responsive web interface
- WebSocket communication for live updates
- Download transcriptions as text files
- Visual status indicators

## Troubleshooting

- Make sure your microphone permissions are enabled
- Check that port 8000 is available
- Ensure all dependencies are installed correctly