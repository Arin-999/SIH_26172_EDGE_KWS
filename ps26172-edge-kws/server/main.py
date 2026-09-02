import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import numpy as np
import os

app = FastAPI()

# Initialize the Whisper model
print("Loading Whisper model...")
model_size = os.getenv("KWS_ASR_MODEL", "tiny.en")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Whisper model loaded.")

@app.websocket("/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Simple protocol: if we receive a single 0xFF byte, it's the end of utterance marker
            if len(data) == 1 and data[0] == 0xFF:
                print("End of utterance marker received. Processing audio...")
                
                # Convert PCM bytes (16kHz, int16, mono) to float32 numpy array
                if len(audio_buffer) > 0:
                    pcm_array = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    segments, info = model.transcribe(pcm_array, beam_size=5)
                    transcript = "".join([segment.text for segment in segments])
                    
                    print(f"Transcript: {transcript}")
                    await websocket.send_json({"type": "transcript", "text": transcript})
                    
                    # Very simple intent handling
                    text_lower = transcript.lower()
                    if "light" in text_lower:
                        await websocket.send_json({"type": "agent_response", "intent": "lights", "action": "toggle"})
                    else:
                        await websocket.send_json({"type": "agent_response", "intent": "unknown", "action": "none"})
                
                audio_buffer.clear()
            else:
                # Accumulate raw PCM data. The ESP32 simplified client will send raw PCM chunks directly.
                audio_buffer.extend(data)
                
    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
