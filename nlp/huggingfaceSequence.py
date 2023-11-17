from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#https://huggingface.co/facebook/wmt21-dense-24-wide-en-x
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

inputs = tokenizer("To translate into a target language, the target language id is forced as the first generated token. To force the target language id as the first generated token, pass the forced_bos_token_id parameter to the generate method.", return_tensors="pt")

# translate English to Chinese
generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("zh")) #max_new_tokens
result=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(result)

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# translator = pipeline("translation", model=model_checkpoint)
# print(translator("Default to expanded threads"))

# from transformers import AutoTokenizer

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# from transformers import AutoModelForSeq2SeqLM
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)