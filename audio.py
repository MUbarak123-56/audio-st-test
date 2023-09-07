import streamlit as st
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
import numpy as np
#import librosa
#import soundfile

checkpoint = "openai/whisper-small.en"  

@st.cache(allow_output_mutation=True)
def model():
    pipe = pipeline("automatic-speech-recognition", model=checkpoint)
    return pipe

audio_bytes = audio_recorder(text="Click Me", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="user", icon_size="3x",)

#processor, model = model()
pipe = model()

if audio_bytes:
    new_audio = st.audio(audio_bytes, format="audio/wav")
    #audio_np = np.array(new_audio)
    data_s16 = np.frombuffer(audio_bytes, dtype=np.int16, count=len(audio_bytes)//2, offset=0)
    float_data = data_s16 * 0.5**15
    audio_input = {}
    audio_input["array"] = np.array(float_data)
    audio_input["sampling_rate"] = 16000
    #input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
    # generate token ids
    #predicted_ids = model.generate(input_features)
    #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    st.write(pipe(audio_input)["text"])
    st.write(audio_input)
    #st.write(audio_bytes)
