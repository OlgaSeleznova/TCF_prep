import torch
import torchaudio
import pyaudio
import numpy as np
import whisper
import queue
from datetime import datetime
import os

class FrenchTranscriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Whisper model
        self.model = whisper.load_model("base").to(self.device)
        print(f"Loaded Whisper model")

        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_recording = False

        # Transcription file
        self.transcription_file = None

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
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
        print("Recording started. Speak in French...")

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("Recording stopped.")

    def transcribe_audio(self, audio_data):
        # Ensure audio is the right format
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Transcribe with Whisper, specifying French
        result = self.model.transcribe(
            audio_data,
            language="fr",
            task="transcribe", fp16=False
        )

        return result["text"].strip()

    def save_transcription(self, text):
        """Save transcription to file"""
        if not self.transcription_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_french_{timestamp}.txt"
            self.transcription_file = open(filename, 'w', encoding='utf-8')
            print(f"Saving transcriptions to: {filename}")

        self.transcription_file.write(f"{text}\n")
        self.transcription_file.flush()

    def real_time_transcribe(self, duration=90):
        """Transcribe audio in real-time chunks"""
        print("starting real-time transcription...")
        self.start_recording()

        try:
            buffer = []
            buffer_duration = 0
            target_buffer_size = self.sample_rate * duration  # 5 seconds of audio

            while self.is_recording:
                try:
                    # Get audio chunk from queue
                    chunk = self.audio_queue.get(timeout=0.1)
                    buffer.extend(chunk)
                    buffer_duration += len(chunk)

                    # When we have enough audio, transcribe it
                    if buffer_duration >= target_buffer_size:
                        audio_array = np.array(buffer)

                        if len(audio_array) > 0:
                            transcription = self.transcribe_audio(audio_array)
                            if transcription:
                                print(f"Transcription: {transcription}")
                                self.save_transcription(transcription)

                        # Keep some overlap for better transcription
                        overlap_size = self.sample_rate * 1  # 1 second overlap
                        buffer = buffer[-overlap_size:] if len(buffer) > overlap_size else []
                        buffer_duration = len(buffer)

                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break

        finally:
            self.stop_recording()
            if self.transcription_file:
                self.transcription_file.close()
                print(f"Transcription saved to: {self.transcription_file.name}")

def main():
    transcriber = FrenchTranscriber()

    print("French Audio Transcriber")
    print("Commands:")
    print("  'start' - Start real-time transcription")
    print("  'quit' - Exit the program")

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == 'start':
            print("Starting real-time transcription... Press Ctrl+C to stop.")
            try:
                transcriber.real_time_transcribe()
                print()
            except KeyboardInterrupt:
                print("\nTranscription stopped by user.")

        elif command == 'quit':
            print("Goodbye!")
            break

        else:
            print("Unknown command. Use 'start' or 'quit'.")

if __name__ == "__main__":
    main()