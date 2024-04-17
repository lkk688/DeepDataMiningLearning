import gradio as gr
# import torch
# import requests
# from PIL import Image
# from torchvision import transforms
from DeepDataMiningLearning.hfvision_inference import MyVisionInference
mycache_dir = r"D:\Cache\huggingface"
# gradio gradio_image.py

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


model_name_or_path = "microsoft/resnet-50"
myinference = MyVisionInference(model_name=model_name_or_path,
                                task="image-classification", model_type="huggingface", cache_dir=mycache_dir)


def predict(inp):
    return myinference(inp)


# with gr.Blocks() as demo:
#     gr.Markdown("Start typing below and then click **Run** to see the output.")
demo = gr.Interface(fn=predict,
              # creates the component and handles the preprocessing to convert that to a PIL image.
              inputs=gr.Image(type="pil"),
              outputs=gr.Label(num_top_classes=3),
              examples=["../sampledata/bus.jpg"]).launch()

if __name__ == "__main__":
    demo.launch(share=True)
