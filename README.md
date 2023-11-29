# TalkGPT
![python](https://img.shields.io/badge/Python-3.9.0%2B-blue)
[![View on Streamlit](https://img.shields.io/badge/Streamlit-View%20on%20Streamlit%20app-ff69b4?logo=streamlit)](https://talk-gpt.streamlit.app/)
[![Read on Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@Mubarak_Ganiyu/talkgpt-voice-integration-with-chatgpt-dfbd02a0ceab)
[![View on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mabrouk/talkgpt)

## Introduction

ChatGPT has become a pivotal assistant enabling users to interact directly with data, facilitating text-based answers from natural language prompts and it recently included voice features for conversation. However, the web-version of ChatGPT does not incorporate the usage of audio for communication. This could prove to be challenging for users with reading or typing difficulties who would like to use the web version of ChatGPT.

One other notable concern for occasional users of ChatGPT is the necessity of a subscription plan to access the optimal version of ChatGPT (GPT-4). While the subscription plan offers substantial benefits for those interested in utilizing multiple plugins and maximizing ChatGPT’s capabilities, it may not be as appealing to those who seek to query GPT-4 for information only occasionally.

Introducing TalkGPT, an app recently developed to address the challenges faced by occasional users who seek a remarkable experience with GPT-4 without a subscription, and who desire voice interaction capabilities. TalkGPT adopts a Pay-As-You-Go approach, inviting users to input an OpenAI API token for connection, subsequently billing them approximately $0.01 for every query sent to GPT-4. The app also employs an Automatic Speech Recognition model from Hugging Face 🤗 for speech transcription and utilizes a Text-to-Speech model from Hugging Face 🤗 to convert text into speech. These Hugging Face models are essential for voice communication.

## References
The two articles below served as an inspiration for building TalkGPT

- How to build a Llama 2 Chatbot ([Link](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/))
- Beginner’s Guide to OpenAI API ([Link](https://blog.streamlit.io/beginners-guide-to-openai-api/))

## Resources
Here is a list of transformer models and a dataset from Hugging Face and OpenAI that contributed towards the development of the app

- OpenAI — Whisper Small (English Version) ([Link](https://huggingface.co/openai/whisper-small.en)) — Speech to text conversion
- Microsoft — SpeechT5 Text-to-Speech ([Link](https://huggingface.co/microsoft/speecht5_tts)) — Text to Speech Conversion
- Microsoft — SpeechT5 HifiGan Vocoder ([Link](https://huggingface.co/microsoft/speecht5_hifigan)) — Speech Conversion into a readable format for audio (log-mel spectogram to waveform)
- Speaker embeddings extracted from CMU ARCTIC ([Link](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors)) — Dataset for replicating real people’s voices when generating audio
- Open AI GPT ([Link](https://platform.openai.com/docs/guides/text-generation)) - Response Generation

## Contact Info
- Mubarak Ganiyu (ganiyubaraq@gmail.com)
