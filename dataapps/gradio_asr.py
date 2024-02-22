
import gradio as gr
from transformers import pipeline
import numpy as np
#gradio gradio_asr.py

#https://www.gradio.app/guides/real-time-speech-recognition

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

#The transcribe function takes a single parameter, audio, which is a numpy array of the audio the user recorded. 
#The pipeline object expects this in float32 format, so we convert it first to float32, and then extract the transcribed text.
def transcribe(audio):
    sr, y = audio
    print("Sampling rate:",sr)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    print("Y shape:", y.shape)
    print("Y type:", y.dtype)

    return transcriber({"sampling_rate": sr, "raw": y})["text"]


demo = gr.Interface(
    transcribe,
    gr.Audio(sources=['upload', 'microphone']),
    "text",
)

demo.launch()
#demo.launch(server_name="localhost")
