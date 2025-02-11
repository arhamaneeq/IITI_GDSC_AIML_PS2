import torch
from transformers import pipeline
import librosa
import io
import numpy as np
from pydub import AudioSegment

def convert_bytes_to_array(audio_bytes):
    try:
        # Convert to bytes
        print(f"Received audio bytes: {len(audio_bytes)}")
        audio_bytes = io.BytesIO(audio_bytes)
        audio_bytes.seek(0)

        # Convert to WAV
        audio = AudioSegment.from_file(audio_bytes)
        if audio is None:
            raise ValueError("pydub failed to convert audio")

        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        # Load into librosa
        audio_array, sample_rate = librosa.load(wav_bytes, sr=16000)
        print(f"Sample Rate: {sample_rate}, Shape: {audio_array.shape}")

        # Return as float32 array
        return np.array(audio_array, dtype=np.float32)

    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None

def transcribe_audio(audio_bytes):
    # Check if GPU is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the model
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )
    
    audio_array = convert_bytes_to_array(audio_bytes)

    prediction = pipe(audio_array, batch_size=1)["text"]
    #print(prediction)
    
    return prediction
