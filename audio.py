import streamlit as st
from audiorecorder import audiorecorder
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch 
#import librosa
#import soundfile

checkpoint = "openai/whisper-small.en"  

@st.cache(allow_output_mutation=True)
def model():
    pipe = pipeline("automatic-speech-recognition", model=checkpoint)
    #processor = WhisperProcessor.from_pretrained(checkpoint)
    #model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    #model.config.forced_decoder_ids = None
    return pipe #processor, model

#audio_bytes = audio_recorder(    text="",    recording_color="#e8b62c",   neutral_color="#6aa36f",   icon_name="user",   icon_size="3x",)

#processor, model = model()

audio = audiorecorder("Click to record", "Click to stop recording")

if not audio.empty():
    # To play audio in frontend:
    st.audio(audio.export().read())  

    # To save audio to a file, use pydub export method:
    audio.export("audio.wav", format="wav")

    # To get audio properties, use pydub AudioSegment properties:
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")


#if audio_bytes:
 #   st.audio(audio_bytes, format="audio/wav")
    #input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
    # generate token ids
    #predicted_ids = model.generate(input_features)
    #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    #st.write(pipe(audio_bytes)["text"])
  #  st.write(audio_bytes)
