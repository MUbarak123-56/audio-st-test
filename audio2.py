import sounddevice as sd
from scipy.io.wavfile import write
import streamlit as st

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
#write('output.wav', fs, myrecording)  # Save as WAV file 
st.audio(myrecording, sample_rate=16000, format="audio/wav")
