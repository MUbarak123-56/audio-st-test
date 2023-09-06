import streamlit as st
from audio_recorder_streamlit import audio_recorder

audio_bytes = audio_bytes = audio_recorder(
    text="",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="user",
    icon_size="6x",
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")