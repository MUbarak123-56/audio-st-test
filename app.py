import streamlit as st
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder
import numpy as np
from scipy.io import wavfile
from io import BytesIO
import openai
from transformers import SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
import os
import base64
import pandas as pd
from openai import OpenAI

st.set_page_config(layout='wide', page_title = "TalkGPT 🎤")
with open("style.css")as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

def add_bg(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;}}
        }}
    </style>
    """,
    unsafe_allow_html=True
    )

#add_bg("ai.png") 

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
    
    return stt_model, processor, vocoder, tts_model

stt_model, processor, vocoder, tts_model = models()

with st.sidebar:
    st.title('Settings')
    if 'OPENAI_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai_api_key = st.secrets['OPENAI_API_TOKEN']
    else:
        openai_api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai_api_key).startswith('sk-') or len(openai_api_key) != 51:
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    st.markdown("<h3 style='text-align: left; color: white;'>Parameters</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: white;'>Choose your parameters below.</h5>", unsafe_allow_html=True)
    
    selected_model = st.selectbox('Choose a GPT model', ['GPT 3.5', 'GPT 4'], index = 1)
    if selected_model == 'GPT 3.5':
        llm = 'gpt-3.5-turbo'
    elif selected_model == 'GPT 4':
        llm = 'gpt-4'
    temp = st.number_input('Temperature', min_value=0.01, max_value=4.0, value=0.1, step=0.01)
    top_percent = st.number_input('Top Percent', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    input_format = st.selectbox("Choose an input format", ["Text", "Audio"], index = 0, disabled=not openai_api_key)
    audio_output = st.selectbox("Do you want audio output?", ["Yes", "No"], index = 0)
    if audio_output == "Yes":
        gender_select = st.selectbox("Choose the gender of your speaker", ["Male", "Female"], index = 1)

openai.api_key = openai_api_key
#person = OpenAI()

@st.cache_data()
def speech_embed():
    embeddings_dataset = load_dataset(dataset_tts, split="validation")
    embeddings_dataset = embeddings_dataset.to_pandas()
    if gender_select == "Male":
        #torch.tensor(list(embded[embded["filename"]=="cmu_us_bdl_arctic-wav-arctic_a0009"]["xvector"]))
        embed_use = torch.tensor(list(embeddings_dataset[embeddings_dataset["filename"]=="cmu_us_bdl_arctic-wav-arctic_a0009"]["xvector"]))
    elif gender_select == "Female":
        embed_use = torch.tensor(list(embeddings_dataset[embeddings_dataset["filename"]=="cmu_us_clb_arctic-wav-arctic_a0144"]["xvector"]))
    return embed_use

speaker_embeddings = speech_embed()


st.markdown("<h1 style='text-align: center; color: gold;'>TalkGPT 🎤</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Welcome to TalkGPT. You can speak to GPT and it will speak back to you.</h3>", unsafe_allow_html=True)

with st.expander("Click to see instructions on how the parameters/settings work"):
    st.markdown("<h8 style='text-align: center; color: white;'>**Enter OpenAI API token**: You can create an OpenAI token [here](https://openai.com/) or learn how to create one by watching this [video](https://www.youtube.com/watch?v=EQQjdwdVQ-M)</h8>", unsafe_allow_html=True)
    st.markdown("<h8 style='text-align: center; color: white;'>**Choose a GPT model**: You can use this parameter to choose between GPT 3.5 and GPT 4.</h8>", unsafe_allow_html=True)
    st.markdown("<h8 style='text-align: center; color: white;'>**Temperature**: You can change this value to transform the creativity of GPT. A high temperature will make GPT too creative to the point that it produces meaningless statements. A very low temperature makes GPT repetitive.</h8>", unsafe_allow_html=True)
    st.markdown("<h8 style='text-align: center; color: white;'>**Top Percent**: This is used to select the top n percent of the predicted next word. This can serve as a way to ensure GPT is likely going to produce words that matter.</h8>", unsafe_allow_html=True)
    st.markdown("<h8 style='text-align: center; color: white;'>**Choose an input format**: You can select between text and audio. If you choose audio, you will have to speak into an audio recorder and if you choose text you will type in your question for GPT.</h8>", unsafe_allow_html=True)
    st.markdown("<h8 style='text-align: center; color: white;'>**Do you want an audio output?**: If you select yes, you will get an audio response from GPT alongside the text response.</h8>", unsafe_allow_html=True)
    st.markdown("<h8 style='text-align: center; color: white;'>**Choose the gender of your speaker**: You can select the gender of your speaker to be a male or female.</h8>", unsafe_allow_html=True)

def tts(input):
    inputs = processor(text=input, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder).cpu().numpy()

    return speech
  
def generate_llm_response():

  use_messages = []
  for i in range(len(st.session_state.messages)):
      use_messages.append({"role": st.session_state.messages[i]["role"], "content": st.session_state.messages[i]["content"]})

  response = openai.chat.completions.create(
    model=llm,
    messages=use_messages,
    temperature = temp,
    top_p = top_percent, 
  )
  return response.choices[0].message.content


if "messages" not in st.session_state.keys():
    st.session_state.messages = []
    initial_system = {"role": "system", "content": "You are a helpful assistant. You must convert all the numerical digits in your generated responses to words because the user cannot read numbers.", "audio":""}
    st.session_state.messages.append(initial_system)
    initial_message = {"role": "assistant", "content": "How may I assist you today?", "audio":""}
    st.session_state.messages.append(initial_message)

with st.chat_message(st.session_state.messages[1]["role"]):
    st.write(st.session_state.messages[1]["content"])
    tts_init1, sampling_rate = tts(st.session_state.messages[1]["content"]), 16000
    st.audio(tts_init1, format='audio/wav', sample_rate=sampling_rate)

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
            
            if audio_output == "Yes":
                if len(message["audio"]) > 100:
                    st.audio(message["audio"], format = "audio/wav", sample_rate=16000)
                else:
                    for i in range(len(message["audio"])):
                        response_no = "Output " + str(i + 1)
                        st.text(response_no)
                        st.audio(message["audio"][i], format='audio/wav', sample_rate=16000)
            else:
                st.text("No Audio Output.")

if input_format == "Text":
    if prompt := st.chat_input("Text Me", disabled=not openai_api_key):
        new_message = {"role": "user", "content": prompt, "audio":""}
        #with st.chat_message(new_message["role"]):
        #    st.write(new_message["content"])
        st.session_state.messages.append(new_message)

elif input_format == "Audio":
    with st.sidebar:
        st.text("Click to Record")
        audio_bytes = audio_recorder(text="", 
                                    recording_color="#e8b62c", 
                                    neutral_color="#6aa36f", 
                                    icon_name="microphone", 
                                    icon_size="6x", 
                                    sample_rate = 16000)
    if audio_bytes:
        
        bytes_io = BytesIO(audio_bytes)
            
        sample_rate, audio_data = wavfile.read(bytes_io)
            
        audio_input = {"array": audio_data[:,0].astype(np.float32)*(1/32768.0), 
                           "sampling_rate": 16000}
        text = str(stt_model(audio_input)["text"])
        new_message = {"role": "user", "content": text, "audio":""}
        #with st.chat_message(new_message["role"]):
        #    st.write(new_message["content"])
        st.session_state.messages.append(new_message)

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
    if audio_output == "Yes":
        if (len(full_response)) >= 500:
            word = full_response
            tot = 0
            collect_response = []
            reuse_words = ""
            next_word = word
            while tot < len(word):
                new_word = next_word[:500]
                good_word = new_word[:len(new_word) - new_word[::-1].find(".")]
                collect_response.append(good_word)
                reuse_words += good_word
                tot += len(good_word)
                next_word = word[tot:]

            #collect_response[-2] = collect_response[-2] + collect_response[-1]
            #collect_response = collect_response[:-1]
            tts_list = []
            for i in range(len(collect_response)):
                response_no = "Output " + str(i + 1)
                st.text(response_no)
                tts_output, sampling_rate = tts(collect_response[i]), 16000
                tts_list.append(tts_output)
                st.audio(tts_output, format='audio/wav', sample_rate=sampling_rate)
            new_message["audio"] = tts_list
        else:
            tts_output, sampling_rate = tts(full_response), 16000
            new_message["audio"] = tts_output
            st.audio(tts_output, format='audio/wav', sample_rate=sampling_rate)
    else:
        st.text("No Audio Output.")
        new_message["audio"] = ""

    st.session_state.messages.append(new_message)
            
def clear_chat_history():
    st.session_state.messages = []
    audio_list = []
    initial_system = {"role": "system", "content": "You are a helpful assistant. You must convert all the numerical digits in your generated responses to words because the user cannot read numbers.", "audio":""}
    st.session_state.messages.append(initial_system)
    initial_message = {"role": "assistant", "content": "How may I assist you today?", "audio":""}
    st.session_state.messages.append(initial_message)
  
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
