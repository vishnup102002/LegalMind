import torch
from transformers import pipeline
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestWhisper")

try:
    logger.info("Initializing whisper-tiny pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cpu"
    )
    logger.info("✓ Pipeline initialized successfully.")
    
    # Generate 1 second of dummy audio (16kHz sampling rate)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    
    logger.info("Transcribing dummy audio...")
    result = pipe(dummy_audio, generate_kwargs={"language": "malayalam"})
    logger.info(f"✓ Result: {result}")
except Exception as e:
    logger.error(f"Whisper initialization or transcription failed: {e}")
