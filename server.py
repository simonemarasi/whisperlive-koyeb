import asyncio
import websockets
import numpy as np
import torch
from faster_whisper import WhisperModel

# Carica Whisper in modalità CPU (puoi cambiare modello: tiny, base, small, etc.)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

async def transcribe(websocket, path):
    print("🔊 Connessione ricevuta...")
    async for message in websocket:
        audio_data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio_data, language="it")
        text = " ".join([s.text for s in segments])
        await websocket.send(text)

start_server = websockets.serve(transcribe, "0.0.0.0", 8000)

print("🎤 Whisper Live avviato su porta 8000...")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
