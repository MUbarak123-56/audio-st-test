import streamlit as st
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
import numpy as np
from scipy.io import wavfile
from io import BytesIO
import openai
#import librosa
#import soundfile

checkpoint = "openai/whisper-small.en"  

openai_api_key = st.text_input('OpenAI API Key', type='password')
if not openai_api_key.startswith('sk-'):
  st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-'):
  audio_bytes = audio_recorder(text="Click Me", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="user", icon_size="1x")

  if audio_bytes:
      new_audio = st.audio(audio_bytes, format="audio/wav")
      bytes_io = BytesIO(audio_bytes)
    
      # Read the file sample rate and data using wavfile
      sample_rate, audio_data = wavfile.read(bytes_io)
      audio_input = {"array": np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0, #audio_data[:,0].astype(np.float32)*(1/32768.0), 
                   "sampling_rate": 16000}
    
      transcription = openai.Audio.transcribe("whisper-1", audio_input)

      st.write(transcription)
