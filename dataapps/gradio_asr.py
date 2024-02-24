
import gradio as gr
from transformers import pipeline
import numpy as np
#gradio gradio_asr.py

#https://www.gradio.app/guides/real-time-speech-recognition
from DeepDataMiningLearning.hfaudio.inference import MyAudioInference
import os
model_name = "facebook/wav2vec2-large-robust-ft-libri-960h"
task = "audio-asr"
device = "cuda"
mycache_dir= r"D:\Cache\huggingface"
os.environ['HF_HOME'] = mycache_dir
inferencemodel = MyAudioInference(model_name, task=task, device=device, cache_dir=mycache_dir)

#transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

#The transcribe function takes a single parameter, audio, which is a numpy array of the audio the user recorded. 
#The pipeline object expects this in float32 format, so we convert it first to float32, and then extract the transcribed text.
def transcribe(audio):
    sr, y = audio
    print("Sampling rate:",sr)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    print("Y shape:", y.shape)
    print("Y type:", y.dtype)

    #return transcriber({"sampling_rate": sr, "raw": y})["text"]
    transcripts = inferencemodel(y, orig_sr=sr)
    return transcripts


demo = gr.Interface(
    transcribe,
    gr.Audio(sources=['upload', 'microphone']),
    "text",
)

demo.launch()
#demo.launch(server_name="localhost")
