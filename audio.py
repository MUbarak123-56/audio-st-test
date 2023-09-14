import streamlit as st
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
import numpy as np
from scipy.io import wavfile
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
#import librosa
#import soundfile

checkpoint = "openai/whisper-small.en"  

@st.cache_resource()
def model():
    pipe = pipeline("automatic-speech-recognition", model=checkpoint)
    #processor = WhisperProcessor.from_pretrained(checkpoint)
    #model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    return pipe#, processor, model

audio_bytes = audio_recorder(text="Click Me", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="user", icon_size="1x", sample_rate = 16000)

pipe = model()
#pipe, processor, model = model()

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if not openai_api_key.startswith('sk-'):
        st.sidebar.warning('Please enter your OpenAI API key!', icon='⚠')
openai.api_key = openai_api_key
def generate_response(input_query):
  #llm = OpenAI(model_name='gpt-4', temperature=0.1, openai_api_key=openai_api_key)
  #llm2 = ChatOpenAI(model_name='gpt-4', temperature=0.1, openai_api_key=openai_api_key)
  #prompt = PromptTemplate(
  #  input_variables=[input_query],
  #  template=input_query,
  #)
    
  #chain = LLMChain(llm=llm, prompt=prompt)
  response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_query},
        #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #{"role": "user", "content": "Where was it played?"}
    ]
  )
  return response["choices"][0]["message"]["content"]

def tts(input):

    return input

if audio_bytes:
    new_audio = st.audio(audio_bytes, format="audio/wav")
    bytes_io = BytesIO(audio_bytes)
    
    sample_rate, audio_data = wavfile.read(bytes_io)
    
    audio_input = {"array": audio_data[:,0].astype(np.float32)*(1/32768.0), 
                   "sampling_rate": 16000}
    st.write(audio_input)
    text = str(pipe(audio_input)["text"])
    st.write(text)
    
    st.info(generate_response(text))
    


