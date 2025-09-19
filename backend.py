from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyaudio
import threading
import queue
from datetime import datetime

app = FastAPI()

class AudioTranscriber:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading Whisper model on {self.device}")
        try:
            model_name = "openai/whisper-base"
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)

            if str(self.device) != "cpu":
                self.model = self.model.half()
        except Exception as e:
            print(f"Failed to load on {self.device}, falling back to CPU: {e}")
            self.device = torch.device("cpu")
            model_name = "openai/whisper-base"
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)

        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1

        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.websocket = None

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self, websocket, loop):
        self.websocket = websocket
        self.loop = loop
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )

        self.is_recording = True
        self.stream.start_stream()

        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self.transcription_loop)
        self.transcription_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()

    def transcribe_audio(self, audio_data):
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        try:
            # Process audio with Whisper processor
            inputs = self.processor(
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )

            input_features = inputs.input_features.to(self.device)

            # Create attention mask
            attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=self.device)

            # Match data type with model
            if str(self.device) != "cpu" and hasattr(self.model, 'dtype'):
                input_features = input_features.half()
            else:
                input_features = input_features.float()

            # Generate transcription using model's generate method for long-form
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    task="transcribe",
                    language="fr",
                    max_new_tokens=440,
                    do_sample=False,
                    num_beams=1
                )

            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription.strip()

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def send_transcription(self, text):
        if self.websocket and text:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    })),
                    self.loop
                )
            except:
                pass

    def transcription_loop(self):
        buffer = []
        buffer_duration = 0
        target_buffer_size = self.sample_rate * 5  # 5 seconds

        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                buffer.extend(chunk)
                buffer_duration += len(chunk)

                if buffer_duration >= target_buffer_size:
                    audio_array = np.array(buffer)

                    if len(audio_array) > 0:
                        transcription = self.transcribe_audio(audio_array)
                        if transcription:
                            # Send transcription via websocket
                            self.send_transcription(transcription)

                    # Keep some overlap to avoid cutting off words
                    overlap_size = self.sample_rate * 3  # 3 seconds overlap
                    buffer = buffer[-overlap_size:] if len(buffer) > overlap_size else []
                    buffer_duration = len(buffer)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
                break

# Global transcriber instance
transcriber = AudioTranscriber()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "start_recording":
                if not transcriber.is_recording:
                    loop = asyncio.get_event_loop()
                    transcriber.start_recording(websocket, loop)
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Recording started"
                    }))

            elif message["type"] == "stop_recording":
                if transcriber.is_recording:
                    transcriber.stop_recording()
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Recording stopped"
                    }))

    except WebSocketDisconnect:
        if transcriber.is_recording:
            transcriber.stop_recording()

@app.get("/")
async def get():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>French Transcription</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <div id="root"></div>
        <script src="/static/bundle.js"></script>
    </body>
    </html>
    """)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")