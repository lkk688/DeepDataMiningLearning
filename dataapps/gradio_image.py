import torch
import requests
from PIL import Image
from torchvision import transforms
from DeepDataMiningLearning.hfvisionmain import vision_inference
mycache_dir= r"D:\Cache\huggingface"
#gradio gradio_image.py

# #https://www.gradio.app/guides/image-classification-in-pytorch
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# # Download human-readable labels for ImageNet.
# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")
# print(labels)

# def predict(inp): #inp: the input image as a PIL image
#   inp = transforms.ToTensor()(inp).unsqueeze(0)
#   with torch.no_grad():
#     prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
#     confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
#   return confidences


import gradio as gr

def predict(inp):
    vision_inference(inp,mycache_dir)

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"), #creates the component and handles the preprocessing to convert that to a PIL image.
             outputs=gr.Label(num_top_classes=3),
             examples=["../sampledata/bus.jpg"]).launch()