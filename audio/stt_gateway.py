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
        self._last_language_hint = None
        
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

    def _process_cloud_groq_asr(self, audio_bytes: bytes, language: str = None):
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
            "model": "whisper-large-v3-turbo",
            "response_format": "verbose_json"
        }
        if language:
            data["language"] = language

        logger.info(f"Uploading audio to Groq Whisper API ({content_type}) with language hint: {language}...")
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

    def _run_asr(self, audio_bytes: bytes, language: str = None):
        """Runs ASR using Groq Cloud API, falling back to local Whisper-tiny on failure."""
        audio_hash = self._get_audio_hash(audio_bytes)
        
        # Check cache — but invalidate if language hint changed (prevents stale Tamil transcriptions)
        if audio_hash == self._last_audio_hash and self._last_language_hint == language:
            logger.info("Serving ASR result from local transcriber cache.")
            return self._last_transcript, self._last_language

        # 1. Try Groq Cloud Whisper API
        try:
            # Default to Malayalam ('ml') if no language hint is passed, since LegalMind is targeted at Indic/Malayalam speech.
            # This prevents Whisper from translating Malayalam speech into English text.
            effective_language = language or "ml"
            
            transcript, lang = self._process_cloud_groq_asr(audio_bytes, language=effective_language)
            
            # Script hallucination guard 1: Malayalam detected but no Malayalam chars
            if lang == "ml" and not effective_language:
                has_malayalam_chars = any(0x0D00 <= ord(c) <= 0x0D7F for c in transcript)
                if not has_malayalam_chars:
                    logger.warning("Detected Malayalam but transcript has no Malayalam chars. Retrying with language='ml'...")
                    transcript, lang = self._process_cloud_groq_asr(audio_bytes, language="ml")
            
            # Script hallucination guard 2: Tamil detected but session is Malayalam
            # Whisper frequently confuses Malayalam and Tamil
            if lang == "ta" and (language == "ml" or language is None):
                has_tamil_chars = any(0x0B80 <= ord(c) <= 0x0BFF for c in transcript)
                has_malayalam_chars = any(0x0D00 <= ord(c) <= 0x0D7F for c in transcript)
                if has_tamil_chars and not has_malayalam_chars:
                    logger.warning(f"Whisper detected Tamil but session language is '{language}'. Malayalam→Tamil confusion suspected. Retrying with explicit language='ml'...")
                    transcript_retry, lang_retry = self._process_cloud_groq_asr(audio_bytes, language="ml")
                    has_ml_retry = any(0x0D00 <= ord(c) <= 0x0D7F for c in transcript_retry)
                    if has_ml_retry or lang_retry == "ml":
                        transcript = transcript_retry
                        lang = "ml"
                        logger.info("✓ Malayalam retry successful — using corrected transcription.")
                    else:
                        lang = "ml"
                        logger.warning("Malayalam retry did not produce Malayalam chars, but forcing lang='ml' based on session.")
                    
            self._last_audio_hash = audio_hash
            self._last_transcript = transcript
            self._last_language = lang
            self._last_language_hint = language
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

    def transcribe(self, audio_bytes: bytes, language: str = None) -> str:
        """Transcribe speech from incoming audio bytes."""
        transcript, _ = self._run_asr(audio_bytes, language=language)
        return transcript

    def detect_language(self, audio_bytes: bytes, language: str = None) -> str:
        """Detect language of the speech in incoming audio bytes."""
        _, lang = self._run_asr(audio_bytes, language=language)
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
