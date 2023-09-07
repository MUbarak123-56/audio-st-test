import streamlit as st
# transformers import pipeline
#from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
#import librosa
#import soundfile
import openai
import os

def transcribe_with_whisper(audio_file):
    response = openai.Audio.transcribe("whisper-1", audio_file, api_key = openai_api_key)
    return response['transcription']

#st.cache(allow_output_mutation=True)
#def model():
#    pipe = pipeline("automatic-speech-recognition", model=checkpoint)
    #processor = WhisperProcessor.from_pretrained(checkpoint)
    #model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    #model.config.forced_decoder_ids = None
#    return pipe #processor, model


#processor, model = model()
st.title("Whisper ASR with Streamlit")
openai_api_key = st.text_input('OpenAI API Key', type='password')
os.environ["OPENAI_API_KEY"] = openai_api_key
audio_bytes = audio_recorder(text="Click Me", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="user", icon_size="3x",)
if audio_bytes:
    new_audio = st.audio(audio_bytes, format="audio/wav")
    #st.write(new_audio)
    transcript = openai.Audio.transcribe("whisper-1", new_audio, api_key = openai_api_key)
    st.write("Transcription:", transcript)
    #input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
    # generate token ids
    #predicted_ids = model.generate(input_features)
    #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    #st.write(pipe(audio_bytes)["text"])
    #st.write(audio_bytes)


