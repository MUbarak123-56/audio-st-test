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
from transformers import SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
from IPython.display import Audio
import os
import base64


st.set_page_config(layout='wide', page_title = "Audio ChatGPT")

checkpoint_stt = "openai/whisper-small.en"  
checkpoint_tts = "microsoft/speecht5_tts"
checkpoint_vocoder = "microsoft/speecht5_hifigan"
dataset_tts = "Matthijs/cmu-arctic-xvectors"

@st.cache_resource()
def models():
    stt_model = pipeline("automatic-speech-recognition", model=checkpoint_stt)
    processor = SpeechT5Processor.from_pretrained(checkpoint_tts)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_tts)
    vocoder = SpeechT5HifiGan.from_pretrained(checkpoint_vocoder)

    #processor = WhisperProcessor.from_pretrained(checkpoint)
    #model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    
    return stt_model, processor, vocoder, tts_model

stt_model, processor, vocoder, tts_model = models()

@st.cache_data()
def speech_embed():
    embeddings_dataset = load_dataset(dataset_tts, split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7301]["xvector"]).unsqueeze(0)
    return speaker_embeddings

speaker_embeddings = speech_embed()

with st.sidebar:
    st.title('GPT Personal Chatbot')
    if 'OPENAI_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai_api_key = st.secrets['OPENAI_API_TOKEN']
    else:
        openai_api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai_api_key).startswith('sk-') or len(openai_api_key) != 51:
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    
    audio_bytes = audio_recorder(text="Click To Record", 
                                 recording_color="#e8b62c", 
                                 neutral_color="#6aa36f", 
                                 icon_name="microphone", 
                                 icon_size="6x", 
                                 sample_rate = 16000)

    st.subheader('Models')
    selected_model = st.sidebar.selectbox('Choose a GPT model', ['GPT 3.5', 'GPT 4'], key='selected_model')
    if selected_model == 'GPT 3.5':
        llm = 'gpt-3.5-turbo'
    elif selected_model == 'GPT 4':
        llm = 'gpt-4'
    temp = st.sidebar.number_input('temperature', min_value=0.01, max_value=4.0, value=0.1, step=0.01)
    top_percent = st.sidebar.number_input('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

openai.api_key = openai_api_key

def generate_response(input_query):

  response = openai.ChatCompletion.create(
    model=llm,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_query},
        #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #{"role": "user", "content": "Where was it played?"}
    ],
    temperature = temp,
    top_p = top_percent
  )
  return response["choices"][0]["message"]["content"]

def tts(input):
    inputs = processor(text=input, return_tensors="pt")
    speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    return speech

def autoplay_audio(data):
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    return st.markdown(md, unsafe_allow_html=True,)
    
#if audio_bytes:
 #   new_audio = st.audio(audio_bytes, format="audio/wav")
 #   bytes_io = BytesIO(audio_bytes)
    
 #   sample_rate, audio_data = wavfile.read(bytes_io)
    
 #   audio_input = {"array": audio_data[:,0].astype(np.float32)*(1/32768.0), 
                   #"sampling_rate": 16000}
 #   st.write(audio_input)
 #   text = str(stt_model(audio_input)["text"])
 #   st.write(text)

    
    #st.info(output)

    #tts_output = np.array(tts(output))
    #st.audio(tts_output, format='audio/wav', sample_rate=16000)

#audio, text = st.tabs(["Text", "Audio"])
  
def clear_chat_history():
    st.session_state.messages = []
    initial_system = {"role": "system", "content": "You are a helpful assistant."}
    st.session_state.messages.append(initial_system)
    initial_message = {"role": "assistant", "content": "How may I assist you today?"}
    st.session_state.messages.append(initial_message)
  
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_llm_response():
  #chain = LLMChain(llm=llm, prompt=prompt)
  use_messages = st.session_state.messages
  #use_messages.append({"role":"user", "content": input_query})
  response = openai.ChatCompletion.create(
    model=llm,
    messages=use_messages,
    temperature = temp,
    top_p = top_percent, 
  )
  return response["choices"][0]["message"]["content"]

if "messages" not in st.session_state.keys():
    st.session_state.messages = []
    initial_system = {"role": "system", "content": "You are a helpful assistant."}
    st.session_state.messages.append(initial_system)
    initial_message = {"role": "assistant", "content": "How may I assist you today?"}
    st.session_state.messages.append(initial_message)

def message_output(message):
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if message["role"] == "assistant":
        with st.chat_message("assistant"):
            use_response = message["content"]
            placeholder = st.empty()
            full_response = ''
            for item in use_response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

message_output(st.session_state.messages[1])


if prompt := st.chat_input("I want sparta"):
    new_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(new_message)
#elif audio_bytes:
    #new_audio = st.audio(audio_bytes, format="audio/wav")
 #   bytes_io = BytesIO(audio_bytes)
    
  #  sample_rate, audio_data = wavfile.read(bytes_io)
    
   # audio_input = {"array": audio_data[:,0].astype(np.float32)*(1/32768.0), 
       #            "sampling_rate": 16000}
   # text = str(stt_model(audio_input)["text"])
   # new_message = {"role": "user", "content": st.chat_input(text)}
   # st.session_state.messages.append(new_message)
     
#input()

for message in st.session_state.messages[2:]:
    message_output(message)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llm_response()
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    new_message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(new_message)
