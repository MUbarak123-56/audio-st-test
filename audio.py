import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import torch 
#import librosa
#import soundfile

checkpoint = "openai/whisper-small.en"  

@st.cache(allow_output_mutation=True)
def model():
    pipe = pipeline("automatic-speech-recognition", model=checkpoint)
    return pipe

audio_bytes = audio_bytes = audio_recorder(
    text="",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="user",
    icon_size="3x",
)

pipe = model()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.write(pipe(audio_bytes)["text"])
