
import gradio as gr
from transformers import pipeline
import numpy as np
#gradio gradio_asr.py

#https://www.gradio.app/guides/real-time-speech-recognition
from DeepDataMiningLearning.hfaudio.inference import MyAudioInference
import os
model_name = "facebook/wav2vec2-large-robust-ft-libri-960h" #"facebook/seamless-m4t-v2-large" #
task = "audio-asr"
mycache_dir= r"D:\Cache\huggingface"
os.environ['HF_HOME'] = mycache_dir
global target_language
target_language = 'eng'
inferencemodel = MyAudioInference(model_name, task=task, target_language=target_language, cache_dir=mycache_dir)
lanaguage_list=inferencemodel.lanaguage_list

#transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

#The transcribe function takes a single parameter, audio, which is a numpy array of the audio the user recorded. 
#The pipeline object expects this in float32 format, so we convert it first to float32, and then extract the transcribed text.
def transcribe(audio):
    print(audio)
    sr, y = audio
    print("Sampling rate:",sr)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    print("Y shape:", y.shape)
    print("Y type:", y.dtype)
    #print(text_options)

    #return transcriber({"sampling_rate": sr, "raw": y})["text"]
    transcripts = inferencemodel(y, orig_sr=sr)
    return transcripts

def update(name, options, radio):
    print(options)
    print(radio)
    target_language = options
    #inferencemodel.target_language = options
    inferencemodel.settarget_lang(options)
    return f"Welcome {name}, you have chosen task {radio} and target language {options}!"

with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = [
            gr.Textbox(placeholder="What is your email?", info="Enter your email address!"),
            gr.Dropdown(
                lanaguage_list, label="Target Language", info="Select the target language!"
            ),
            gr.Radio(["Transcript", "Translation"], label="Choose Task", info="Choose the task you want to do!"),
        ]
        out = gr.Textbox(info="Your current setups") #"text",
    btn = gr.Button("Setup Task")
    btn.click(fn=update, inputs=inp, outputs=out)
    gr.Interface(
        transcribe,
        gr.Audio(sources=['upload', 'microphone']),
        "text",
    )

# with gr.Blocks() as demo:
#     gr.Markdown("Start typing below and then click **Run** to see the output.")
#     with gr.Row():
#         inp = gr.Textbox(placeholder="What is your name?")
#         out = gr.Textbox()
#     btn = gr.Button("Run")
#     btn.click(fn=update, inputs=inp, outputs=out)
#     text_options = gr.Dropdown(
#             ["cat", "dog", "bird"], label="Animal", info="Will add more animals later!"
#         ),
#     gr.Interface(
#         transcribe,
#         gr.Audio(sources=['upload', 'microphone']),
#         "text",
#     )

# demo = gr.Interface(
#     transcribe,
#     gr.Audio(sources=['upload', 'microphone']),
#     "text",
# )
if __name__ == "__main__":
    demo.launch()
#demo.launch()
#share=True
#demo.launch(server_name="localhost")
