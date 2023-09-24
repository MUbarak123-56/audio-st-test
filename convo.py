import streamlit as st
#from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
import openai
from datasets import load_dataset
import torch
from IPython.display import Audio
import os
import base64

#Audio(speech, rate=16000)
#import librosa
st.set_page_config(layout='wide', page_title = "Custom-made ChatGPT")

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
    
    st.subheader('Models')
    selected_model = st.sidebar.selectbox('Choose a GPT model', ['GPT 3.5', 'GPT 4'], key='selected_model')
    if selected_model == 'GPT 3.5':
        llm = 'gpt-3.5-turbo'
    elif selected_model == 'GPT 4':
        llm = 'gpt-4'
    temp = st.sidebar.number_input('temperature', min_value=0.01, max_value=4.0, value=0.1, step=0.01)
    top_percent = st.sidebar.number_input('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    
openai.api_key = openai_api_key
  
def clear_chat_history():
    st.session_state.messages = []
    initial_system = {"role": "system", "content": "You are a helpful assistant."}
    st.session_state.messages.append(initial_system)
    initial_message = {"role": "assistant", "content": "How may I assist you today?"}
    st.session_state.messages.append(initial_message)
  
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_response():
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

if prompt := st.chat_input():
    new_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(new_message)

for message in st.session_state.messages[2:]:
    message_output(message)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response()
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    new_message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(new_message)
