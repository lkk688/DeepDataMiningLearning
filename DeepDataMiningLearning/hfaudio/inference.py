import librosa
from tqdm.auto import tqdm
import torch
import torchaudio
import torch.nn.functional as F
import math
import collections
import numpy as np
import random

from DeepDataMiningLearning.hfaudio.hfdata import saveaudio_tofile
from DeepDataMiningLearning.hfaudio.hfmodels import loaddefaultmodel_fromname

def settarget_lang(model, processor, target_lang='eng'):
    processor.tokenizer.set_target_lang(target_lang) #"cmn-script_simplified"
    model.load_adapter(target_lang)


def asrinference_path(datasample, model, samplerate=16_000, processor=None, device='cuda'):
    if isinstance(datasample, str):#batch["path"] get numpyarray
        datasamples, sampling_rate = librosa.load(datasample, sr=samplerate)
        batchdecode=False
    elif isinstance(datasample, list):
        datasamples=[]
        for sample in datasample:
            onespeech, sampling_rate = librosa.load(sample, sr=samplerate)
            datasamples.append(onespeech)
        batchdecode=True
    
    inputs = processor(datasamples, sampling_rate=samplerate, return_tensors="pt", padding=True)
    print("Input:", type(inputs))

    model=model.to(device)
    inputs=inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    print("Output logits shape:", outputs.shape)
    if batchdecode:
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
    else:
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)
    return transcription

from transformers import pipeline
def inferencesample(datasample, task, model, usepipeline=True, feature_extractor=None, processor=None, device='cuda', task_column='audio', target_column='text'):
    sample = datasample[task_column] #raw_datasets["train"][0][task_column]
    print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")

    saveaudio_tofile(sample['array'], "./output", sample["sampling_rate"], filename="save_example.wav")
    #loadaudio_fromfile(filepath=sample["path"], plotfig=True)

    if usepipeline:
        mypipeline = pipeline(
            task, #"audio-classification",
            model=model, #"anton-l/xtreme_s_xlsr_300m_minds14",
        )
        #sample = datasample[columnname]
        result = mypipeline(sample)#sample can be audio (sample["audio"]) or path (minds[0]["path"])
        print(result)
    else:
        #sample=datasample[columnname]
        #sample["audio"]["array"]
        print(feature_extractor.sampling_rate)
        print(sample['sampling_rate']) #dataset.features["audio"].sampling_rate 
        
        if isinstance(model, str):
            model, feature_extractor, tokenizer, processor = loaddefaultmodel_fromname(modelname=model, task=task)
        
        inputs = feature_extractor(sample["array"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")#size did not change
        # inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        print(f"inputs keys: {list(inputs.keys())}")#[1, 49562]
        input_values=inputs['input_values'].numpy()#(1, 48640)
        print(
            f"Mean: {np.mean(input_values):.3}, Variance: {np.var(input_values):.3}"
        )
        if processor:
            inputs2 = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
            input_values=inputs2['input_values'].numpy()
            print(
                f"Mean: {np.mean(input_values):.3}, Variance: {np.var(input_values):.3}"
            )
            #test tokenizer
            #tokenids = processor.tokenizer(datasample[target_column]).input_ids
            #decoded_str = processor.tokenizer.batch_decode(tokenids, group_tokens=False)

        model=model.to(device)
        inputs=inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        #Get the class with the highest probability
        predicted_ids=torch.argmax(logits, dim=-1) #[1, 45]->33; [1, 113, 32]->[1, 113]

        if task == "audio-classification":
            predicted_class_ids = predicted_ids.item() #.item() moves the scalar data to CPU
            #use the model’s id2label mapping to convert it to a label:
            result = model.config.id2label[str(predicted_class_ids)]
            print(result)
            #list all classes
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            allclasses = [{"class": model.config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
        elif task == "audio-asr":
            # transcribe speech
            transcription = processor.batch_decode(predicted_ids)
            result = transcription[0]
            print(transcription[0])
            # retrieve word stamps (analogous commands for `output_char_offsets`)
            outputs = processor.tokenizer.decode(np.squeeze(predicted_ids), output_word_offsets=True)
            print(outputs["text"])
            text=datasample[target_column]
            # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate 
            time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
            word_offsets = [
                {
                    "word": d["word"],
                    "start_time": round(d["start_offset"] * time_offset, 2),
                    "end_time": round(d["end_offset"] * time_offset, 2),
                }
                for d in outputs.word_offsets
            ]
            print(word_offsets[:3])

    return result
    
def inferenceaudiopath(data_path, task, model, usepipeline=True, feature_extractor=None, device='cuda', columnname='audio'):
    if isinstance(data_path, str):
        speech_array, _sampling_rate = torchaudio.load(data_path)
        resampler = torchaudio.transforms.Resample(feature_extractor.sampling_rate)
        audio_np = resampler(speech_array).squeeze().numpy()
    #elif isinstance(data_path, np.array):
    else:
        audio_np = data_path
    
    features = feature_extractor(audio_np, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    
    logitsmax=torch.argmax(logits) #[1, 45]->33
    predicted_class_ids = logitsmax.item() #.item() moves the scalar data to CPU
    #use the model’s id2label mapping to convert it to a label:
    result = model.config.id2label[str(predicted_class_ids)]
    print(result)
    
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Target": model.config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    print(f"file path: {data_path} \t predict: {result} \t score:{scores[predicted_class_ids]} ")

    return result, outputs

#https://huggingface.co/facebook/seamless-m4t-v2-large
#eng	English
#cmn	Mandarin Chinese
def seamlessm4t_inference(model_name="facebook/seamless-m4t-v2-large", dataset=None, device="cuda:0", cache_dir="", target_language='eng'):
    #model_name="facebook/seamless-m4t-v2-large"
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    import scipy

    
    model = SeamlessM4Tv2Model.from_pretrained(model_name, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast = False)

    model = model.to(device)
    sample_rate = dataset[0]["audio"]["sampling_rate"]
    audio = dataset[0]["audio"]["array"]

    inputs = processor(audios=audio, sampling_rate=sample_rate, return_tensors="pt").to(device) #torch.Size([1, 314, 160])

    #audio to audio
    audio_array_from_audio = model.generate(**inputs, tgt_lang=target_language)[0].cpu().numpy().squeeze()

    scipy.io.wavfile.write("./output/seamless_m4t_out.wav", rate=sample_rate, data=audio_array_from_audio) #

    #audio to text
    output_tokens = model.generate(**inputs, tgt_lang=target_language, generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    print(f"Translation from audio: {translated_text_from_audio}")


#select the correct forced_bos_token_id given your choosen language id
MAPPING = {
    "de": 250003,
    "tr": 250023,
    "fa": 250029,
    "sv": 250042,
    "mn": 250037,
    "zh": 250025,
    "cy": 250007,
    "ca": 250005,
    "sl": 250052,
    "et": 250006,
    "id": 250032,
    "ar": 250001,
    "ta": 250044,
    "lv": 250017,
    "ja": 250012,
}

def wave2vec2seq_inference(model_name, dataset, device="cuda:0", cache_dir="", target_language='en'):
    #https://huggingface.co/facebook/wav2vec2-xls-r-1b-en-to-15
    #"facebook/wav2vec2-xls-r-1b-en-to-15" "facebook/wav2vec2-xls-r-300m-en-to-15"
    #Facebook's Wav2Vec2 XLS-R fine-tuned for Speech Translation
    import torch
    from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
    model = SpeechEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Speech2Text2Processor.from_pretrained(model_name, cache_dir=cache_dir)#, use_fast = False)
    forced_bos_token = MAPPING[target_language]

    model = model.to(device)
    sample_rate = dataset[0]["audio"]["sampling_rate"]
    audio = dataset[0]["audio"]["array"]

    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)#, forced_bos_token_id=forced_bos_token)
    transcription = processor.batch_decode(generated_ids)
    print(transcription)


if __name__ == "__main__":
    from hfdata import gettestdata
    from datasets import DatasetDict
    from hfutil import deviceenv_set

    USE_HPC = True
    if USE_HPC:
        mycache_dir = deviceenv_set(True, data_path='/data/cmpe249-fa23/Huggingfacecache')
    else:
        mycache_dir = "/DATA10T/Cache"

    #model_name = "facebook/wav2vec2-xls-r-300m"
    output_dir = './output'

    dataset_name = "mozilla-foundation/common_voice_11_0"
    language = 'en'
    dataset_test = gettestdata(dataset_name, language=language, split="test", sampling_rate=16000, mycache_dir=mycache_dir, streaming=False, samples = 100, task_column="audio", target_column="sentence")
    #raw_datasets = DatasetDict()
    #raw_datasets["train"] = dataset_test

    model_name = "facebook/s2t-wav2vec2-large-en-de"#"facebook/wav2vec2-xls-r-1b-en-to-15"
    wave2vec2seq_inference(model_name, dataset_test, cache_dir=mycache_dir, target_language='de')

    seamlessm4t_inference(model_name="facebook/seamless-m4t-v2-large", dataset=dataset_test, device="cuda:0", cache_dir=mycache_dir, target_language='eng')

    


