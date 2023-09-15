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

#Audio(speech, rate=16000)
#import librosa
#import soundfile

checkpoint_stt = "openai/whisper-small.en"  
checkpoint_tts = "microsoft/speecht5_tts"
checkpoint_vocoder = "microsoft/speecht5_hifigan"
dataset_tts = "Matthijs/cmu-arctic-xvectors"

@st.cache_resource()
def model():
    stt_model = pipeline("automatic-speech-recognition", model=checkpoint_stt)
    processor = SpeechT5Processor.from_pretrained(checkpoint_tts)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_tts)
    vocoder = SpeechT5HifiGan.from_pretrained(checkpoint_vocoder)
    embeddings_dataset = load_dataset(dataset_tts, split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    #processor = WhisperProcessor.from_pretrained(checkpoint)
    #model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    
    return stt_model, processor, vocoder, tts_model, speaker_embeddings

stt_model, processor, vocoder, tts_model, speaker_embeddings = model()
#pipe, processor, model = model()

audio_bytes = audio_recorder(text="Click Me", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="user", icon_size="1x", sample_rate = 16000)

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
    inputs = processor(text=input, return_tensors="pt")
    speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    return speech

if audio_bytes:
    new_audio = st.audio(audio_bytes, format="audio/wav")
    bytes_io = BytesIO(audio_bytes)
    
    sample_rate, audio_data = wavfile.read(bytes_io)
    
    audio_input = {"array": audio_data[:,0].astype(np.float32)*(1/32768.0), 
                   "sampling_rate": 16000}
    st.write(audio_input)
    text = str(stt_model(audio_input)["text"])
    st.write(text)

    output = generate_response(text)
    st.info(output)

    tts_output = np.array(tts(output))
    #st.audio(tts_output, sample_rate = 16000, format ="audio/wav")
    Audio(tts_output, rate = 16000)

