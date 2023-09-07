import streamlit as st
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
import numpy as np
from scipy.io import wavfile
from io import BytesIO
#import librosa
#import soundfile

checkpoint = "openai/whisper-small.en"  

@st.cache_resource()
def model():
    pipe = pipeline("automatic-speech-recognition", model=checkpoint)
    processor = WhisperProcessor.from_pretrained(checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    return pipe#, processor, model

audio_bytes = audio_recorder(text="Click Me", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="user", icon_size="1x", sample_rate = 16_000)

pipe = model()
#pipe, processor, model = model()

if audio_bytes:
    new_audio = st.audio(audio_bytes, format="audio/wav")
    bytes_io = BytesIO(audio_bytes)
    
    # Read the file sample rate and data using wavfile
    sample_rate, audio_data = wavfile.read(bytes_io)
    #audio_input = {"array": np.frombuffer(audio_data, np.int16).flatten().astype(np.float32)/32768.0, #audio_data[:,0].astype(np.float32)*(1/32768.0), 
    #"sampling_rate": sample_rate}
    
    audio_input = {"array": audio_data[:,0].astype(np.float32)*(1/32768.0), #audio_data[:,0].astype(np.float32)*(1/32768.0), 
                   "sampling_rate": sample_rate}
    st.write(audio_input)
    st.write(pipe(audio_input)["text"])
    
    #input_features = processor(audio_input["array"], sampling_rate=16000, return_tensors="pt").input_features 
    
    #predicted_ids = model.generate(input_features)
    # decode token ids to text
    #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    
    #st.write(transcription)
    
    
