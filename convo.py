import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
from datasets import load_dataset
import torch
from IPython.display import Audio
import os
import base64

#Audio(speech, rate=16000)
#import librosa
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
    model=llm,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_query},
        #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #{"role": "user", "content": "Where was it played?"}
    ]
  )
  return response["choices"][0]["message"]["content"]
  
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
  
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
