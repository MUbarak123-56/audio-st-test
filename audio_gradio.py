import gradio as gr
from transformers import pipeline
import torch 
import librosa
import soundfile

checkpoint = "mabrouk/whisper-small-yo"  

pipe = pipeline("automatic-speech-recognition", model=checkpoint)

def transcribe(Microphone, File_Upload):
    warn_output = ""
    if (Microphone is not None) and (File_Upload is not None):
        warn_output = "WARNING. You have uploaded an audio file and used the microphone." \
                      "The recorded file from the microphone will be used and the uploaded audio will be discarded. \n"
        file = Microphone
    elif (Microphone is None) and (File_Upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"
    elif Microphone is not None:
        file = Microphone
    else:
        file = File_Upload

    text = pipe(file)["text"]
    return warn_output + text

iface = gr.Interface(
    fn = transcribe, 
    inputs = [
        gr.inputs.Audio(source = "microphone", type = "filepath", optional=True),
        gr.inputs.Audio(source = "upload", type = "filepath", optional=True)
    ],
    outputs = "text",
    layout = "horizontal",
    theme = "huggingface",
    title = "Whisper Speech Recognition Demo - Yoruba",
    description = f"Demo for speech recognition using the fine-tuned checkpoint: [{checkpoint}](https://huggingface.co/{checkpoint}).",
    allow_flagging = "never"
)

iface.launch(enable_queue = True)
