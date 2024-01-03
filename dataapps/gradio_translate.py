#https://www.gradio.app/guides/quickstart
#pip install gradio
#gradio app.py
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline #pip install transformers
import torch
import os
mycache_dir="D:/Cache"
os.environ['HF_HOME'] = mycache_dir
modelcache_dir=os.path.join(mycache_dir, "models")
os.environ['TRANSFORMERS_CACHE'] = mycache_dir
datasetcache_dir=os.path.join(mycache_dir, "datasets")
os.environ['HF_DATASETS_CACHE'] = mycache_dir #os.path.join(mycache_dir,"datasets") 

# this model was loaded from https://hf.co/models
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", cache_dir=modelcache_dir)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", cache_dir=modelcache_dir)
device = 0 if torch.cuda.is_available() else -1
#https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
#LANGS = ["ace_Arab", "eng_Latn", "fra_Latn", "spa_Latn"]
Alllangdict={'English':'English', 'Chinese':'zho_Hans', 'Chinese (Traditional)':'zho_Hant', 'German':'deu_Latn', 'deu_Latn':'ell_Grek', 'French':'fra_Latn', 'Irish':'gle_Latn', \
             'Hindi':'hin_Deva', 'Italian': 'ita_Latn', 'Japanese': 'ita_Latn', 'Korean':'kor_Hang', 'Malayalam':'mal_Mlym', \
            'Portuguese':'por_Latn', 'Romanian':'ron_Latn', 'Russian':'rus_Cyrl', 'Spanish':'spa_Latn', 'Swedish':'swe_Latn',\
                'Vietnamese':'vie_Latn'}
LANGS = list(Alllangdict.keys()) #["eng_Latn", "zho_Hans"]

def translate(text, src_lang, tgt_lang):
    """
    Translate the text from source lang to target lang
    """
    src_langcode=Alllangdict[src_lang]
    tgt_langcode=Alllangdict[tgt_lang]
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_langcode, tgt_lang=tgt_langcode, max_length=400, device=device)
    result = translation_pipeline(text)
    return result[0]['translation_text']

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.components.Textbox(label="Text", lines=4, placeholder="Translate..."),
        gr.components.Dropdown(label="Source Language", choices=LANGS, value=LANGS[0]),
        gr.components.Dropdown(label="Target Language", choices=LANGS, value=LANGS[0]),
    ],
    outputs=["text"],
    examples=[["Building a translation demo with Gradio is so easy!", "en", "zh"]],
    cache_examples=True,
    title="Translation Demo",
    description="This demo is a simplified version of the original [NLLB-Translator](https://huggingface.co/spaces/Narrativaai/NLLB-Translator) space"
)

    
if __name__ == "__main__":
    demo.launch(show_api=False)   