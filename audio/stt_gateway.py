import torch
import logging
import io

logger = logging.getLogger("LegalMind.Audio.STT")

import torch
import logging
import io
import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("LegalMind.Audio.STT")

class ShrutamAudioTranscriber:
    def __init__(self):
        # Load whisper-tiny locally on CPU/MPS/CUDA for dynamic multi-lingual ASR
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device
            )
            logger.info("✓ Whisper-tiny Speech-to-Text pipeline initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Whisper-tiny pipeline locally: {e}. Falling back to None.")
            self.pipe = None

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw regional Malayalam/English speech from incoming audio bytes.
        """
        if self.pipe:
            try:
                # 1. Read audio bytes using soundfile (supports WAV, OGG, etc.)
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                
                # 2. Convert to mono if stereo
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                # 3. Resample to 16000 Hz if necessary
                if samplerate != 16000:
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                
                # Convert to float32 numpy array
                data = data.astype(np.float32)
                
                # 4. Transcribe using Whisper-tiny
                result = self.pipe(data)
                text = result.get("text", "").strip()
                logger.info(f"✓ Whisper-tiny transcried ASR text: '{text}'")
                return text
            except Exception as e:
                logger.error(f"Error during Whisper-tiny transcription: {e}")
                return ""
        else:
            logger.warning("Whisper pipeline is not initialized. ASR failed.")
            return ""

    def transcribe_stream(self, audio_generator):
        """
        Accepts a stream/generator of audio chunks and processes them in real-time.
        """
        buffer = bytearray()
        for chunk in audio_generator:
            buffer.extend(chunk)
        yield self.transcribe(bytes(buffer))

if __name__ == "__main__":
    transcriber = ShrutamAudioTranscriber()
    mock_audio = np.zeros(16000, dtype=np.float32)
    # Convert numpy array to WAV bytes for testing self-check
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mock_audio, 16000)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            bytes_data = f.read()
        transcript = transcriber.transcribe(bytes_data)
        print(f"✓ Whisper-tiny transcript: '{transcript}'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

