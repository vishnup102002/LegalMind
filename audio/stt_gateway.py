import torch
import logging
import io
import os
import requests
import hashlib
import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("LegalMind.Audio.STT")

class ShrutamAudioTranscriber:
    def __init__(self):
        # Cache to prevent duplicate cloud API calls when calling transcribe() and detect_language() in sequence
        self._last_audio_hash = None
        self._last_transcript = ""
        self._last_language = "en"
        
        # Load local whisper-tiny as fallback ASR
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device
            )
            logger.info("✓ Whisper-tiny local Speech-to-Text pipeline initialized as fallback.")
        except Exception as e:
            logger.warning(f"Failed to load Whisper-tiny pipeline locally: {e}. Falling back to None.")
            self.pipe = None

    def _get_audio_hash(self, audio_bytes: bytes) -> str:
        return hashlib.md5(audio_bytes).hexdigest()

    def _process_cloud_groq_asr(self, audio_bytes: bytes):
        """Invoke Groq Cloud Whisper API to transcribe and detect language."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or not groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is not configured.")

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}"
        }
        
        # Twilio sends voice notes as OGG (Opus) files, we can upload them directly
        # Determine basic mime/extension. Default to ogg/opus, but can fall back
        filename = "voice_note.ogg"
        content_type = "audio/ogg"
        
        # Simple header check for WAV
        if audio_bytes.startswith(b"RIFF"):
            filename = "voice_note.wav"
            content_type = "audio/wav"

        files = {
            "file": (filename, io.BytesIO(audio_bytes), content_type)
        }
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json"
        }

        logger.info(f"Uploading audio to Groq Whisper API ({content_type})...")
        response = requests.post(url, headers=headers, files=files, data=data, timeout=20)
        
        if response.status_code != 200:
            raise RuntimeError(f"Groq Whisper API returned error {response.status_code}: {response.text}")
            
        res_json = response.json()
        transcript = res_json.get("text", "").strip()
        raw_lang = res_json.get("language", "english").lower()
        
        # Convert full language name to ISO code
        if "malayalam" in raw_lang:
            lang = "ml"
        elif "hindi" in raw_lang:
            lang = "hi"
        elif "tamil" in raw_lang:
            lang = "ta"
        elif "kannada" in raw_lang:
            lang = "kn"
        elif "telugu" in raw_lang:
            lang = "te"
        else:
            lang = "en"
            
        logger.info(f"✓ Groq Whisper API successfully transcribed: '{transcript[:60]}...' (detected language: '{lang}' from raw: '{raw_lang}')")
        return transcript, lang

    def _run_asr(self, audio_bytes: bytes):
        """Runs ASR using Groq Cloud API, falling back to local Whisper-tiny on failure."""
        audio_hash = self._get_audio_hash(audio_bytes)
        
        # Check cache
        if audio_hash == self._last_audio_hash:
            logger.info("Serving ASR result from local transcriber cache.")
            return self._last_transcript, self._last_language

        # 1. Try Groq Cloud Whisper API (Highly accurate and fast for Malayalam)
        try:
            transcript, lang = self._process_cloud_groq_asr(audio_bytes)
            self._last_audio_hash = audio_hash
            self._last_transcript = transcript
            self._last_language = lang
            return transcript, lang
        except Exception as cloud_err:
            logger.warning(f"Groq Cloud Whisper API failed: {cloud_err}. Falling back to local Whisper-tiny.")

        # 2. Local Whisper-tiny Fallback
        if self.pipe:
            try:
                # Read audio bytes using soundfile
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if samplerate != 16000:
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                data = data.astype(np.float32)
                
                result = self.pipe(data, return_timestamps=False)
                text = result.get("text", "").strip()
                
                # Detect language using Unicode analysis
                if any(0x0D00 <= ord(c) <= 0x0D7F for c in text):
                    lang = "ml"
                elif any(0x0900 <= ord(c) <= 0x097F for c in text):
                    lang = "hi"
                elif any(0x0B80 <= ord(c) <= 0x0BFF for c in text):
                    lang = "ta"
                elif any(0x0C80 <= ord(c) <= 0x0CFF for c in text):
                    lang = "kn"
                elif any(0x0C00 <= ord(c) <= 0x0C7F for c in text):
                    lang = "te"
                else:
                    lang = "en"
                
                logger.info(f"✓ Fallback Whisper-tiny transcribed: '{text}' (lang={lang})")
                self._last_audio_hash = audio_hash
                self._last_transcript = text
                self._last_language = lang
                return text, lang
            except Exception as local_err:
                logger.error(f"Fallback local Whisper-tiny transcription failed: {local_err}")
                
        return "", "en"

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe speech from incoming audio bytes."""
        transcript, _ = self._run_asr(audio_bytes)
        return transcript

    def detect_language(self, audio_bytes: bytes) -> str:
        """Detect language of the speech in incoming audio bytes."""
        _, lang = self._run_asr(audio_bytes)
        return lang

    def transcribe_stream(self, audio_generator):
        """Accepts a stream/generator of audio chunks and processes them."""
        buffer = bytearray()
        for chunk in audio_generator:
            buffer.extend(chunk)
        yield self.transcribe(bytes(buffer))

if __name__ == "__main__":
    transcriber = ShrutamAudioTranscriber()
    mock_audio = np.zeros(16000, dtype=np.float32)
    # Convert numpy array to WAV bytes for testing self-check
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mock_audio, 16000)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            bytes_data = f.read()
        transcript = transcriber.transcribe(bytes_data)
        print(f"✓ Whisper transcript: '{transcript}'")
        lang = transcriber.detect_language(bytes_data)
        print(f"✓ Detected language: '{lang}'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
