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
from transformers import AutoProcessor, SeamlessM4Tv2Model
import scipy

from DeepDataMiningLearning.hfaudio.hfdata import gettestdata
from datasets import DatasetDict
from DeepDataMiningLearning.hfaudio.hfutil import deviceenv_set, download_youtube, clip_video, load_json, get_device
import os
# from huggingface_hub import login
# login()
import pandas as pd
import moviepy.editor as mp 

def settarget_lang(model, processor, target_lang='eng'):
    processor.tokenizer.set_target_lang(target_lang) #"cmn-script_simplified"
    model.load_adapter(target_lang)

def saveaudioarr2file(audioarr, filename="audio_out.wav", sample_rate=16000):
    scipy.io.wavfile.write(filename, rate=sample_rate, data=audioarr) # audio_array_from_audio

def downsamping(audio, orig_freq, new_freq):
    # you need to convert the audio from a numpy array to a torch tensor first
    audio = torch.tensor(audio)
    if orig_freq != new_freq:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=new_freq)
    return audio
    #new_freq=model.config.sampling_rate

def preprocess_audio(datasample, samplerate=16_000, orig_sr=44100, maxseconds=10):
    sampling_rate = samplerate
    if isinstance(datasample, str):#batch["path"] get numpyarray
        datasamples, sampling_rate = librosa.load(datasample, sr=samplerate)
        # datasamples = librosa.resample(datasamples, orig_sr=orig_sr, target_sr=samplerate)
        batchdecode=False
    elif isinstance(datasample, np.ndarray):
        if datasample.ndim>1: #stereo
            #print(datasample.shape) 
            #(455112, 2)
            if datasample.shape[0]>2 and datasample.shape[0]>datasample.shape[1]:
                datasample = np.transpose(datasample) #=>(2,n)
            audio_array = librosa.to_mono(datasample)
        else:
            audio_array = datasample
        datasamples = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=samplerate)
        batchdecode=False
    elif isinstance(datasample, list):
        if len(datasample)> 0 and isinstance(datasample[0], np.ndarray):
            datasamples=[]
            for sample in datasample:
                onespeech = librosa.resample(sample, orig_sr=orig_sr, target_sr=samplerate)
                datasamples.append(onespeech)
        else:
            datasamples=[]
            for sample in datasample:
                onespeech, sampling_rate = librosa.load(sample, sr=samplerate)
                # onespeech = librosa.resample(onespeech, orig_sr=orig_sr, target_sr=samplerate)
                datasamples.append(onespeech)
        batchdecode=True
    #cut the numpy array based on maximum allowable length
    if len(datasamples) > maxseconds * samplerate:
        datasamples = datasamples[:maxseconds * samplerate]
    datasamples = torch.tensor(datasamples)
    return datasamples, sampling_rate, batchdecode

#"audio-asr" "audio-classification"
class MyAudioInference():
    def __init__(self, model_name, task="audio-asr", target_language='eng', cache_dir="./output", gpuid='0', combineoutput=False, generative=False) -> None:
        self.target_language = target_language #"cmn"
        self.cache_dir = cache_dir
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device, useamp = get_device(gpuid=gpuid, useamp=False)
        self.task = task
        self.model_name = model_name
        if isinstance(model_name, str) and "wav2vec2" in model_name.lower():
            self.model, self.feature_extractor, self.tokenizer, self.processor = loaddefaultmodel_fromname(modelname=model_name, task=task, cache_dir=cache_dir)
            self.generative = False
        elif isinstance(model_name, torch.nn.Module):
            self.generative = generative
        elif "seamless" in model_name.lower():
            self.model = SeamlessM4Tv2Model.from_pretrained(model_name, cache_dir=cache_dir)
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast = False)
            self.generative = True
        self.model=self.model.to(self.device)
        self.combineoutput = combineoutput
        self.languagedicts = {"English": "eng", "Chinese": "cmn", "Japanese": "jpn", "Korean": "kor", "French": "fra", "German": "deu", "Spanish": "spa", "Italian": "ita", "Portuguese": "por", "Russian": "rus", "Vietnamese": "vie", "Cantonese": "yue", "Tagalog": "tgl"}
        self.lanaguage_list = list(self.languagedicts.keys())

    def settarget_lang(self, target_lang='English'):
        self.target_language = self.languagedicts[target_lang] #"
        print("Updated target language:", self.target_language)

    def __call__(self, audiopath, orig_sr=44100) -> torch.Any:
        self.model=self.model.to(self.device)

        if self.task == "audio-asr":
            samplerate = self.processor.feature_extractor.sampling_rate
            transcription = asrinference_path(audiopath, self.model, samplerate=samplerate, processor=self.processor, orig_sr=orig_sr, src_lang=None, tgt_lang=self.target_language, device=self.device, generative=self.generative)
            if self.combineoutput and isinstance(transcription, list):
                output = ""
                for text in transcription:
                    output += text + " "
            else:
                output = transcription
            return output
        
def asrinference_path(datasample, model, samplerate=16_000, orig_sr=44100, src_lang=None, tgt_lang='eng', processor=None, device='cuda', generative=False):
    datasamples, samplerate, batchdecode = preprocess_audio(datasample, samplerate=samplerate, orig_sr=orig_sr, maxseconds=10)
    
    ## Tokenize the audio
    if generative is False:
        #https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/processing_wav2vec2.py 
        #(audio->feature_extractor, text->tokenizer)
        inputs = processor(audio=datasamples, sampling_rate=samplerate, return_tensors="pt", padding=True)
    else: #Seamless
        #https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t/processing_seamless_m4t.py
        #audios->feature_extractor
        #text->tokenizer (tgt_lang, src_lang)
        inputs = processor(audios=datasamples, src_lang=src_lang, tgt_lang=tgt_lang, sampling_rate=samplerate, return_tensors="pt", padding=True)
    #['input_features' [1, 431, 160], 'attention_mask']
    #print("Input:", type(inputs))

    #model=model.to(device)
    inputs=inputs.to(device)

    if generative is False:
        with torch.no_grad():
            outputs = model(**inputs).logits #inputs key=input_values
        #print("Output logits shape:", outputs.shape) #[3, 499, 32]
        if batchdecode:
            predicted_ids = torch.argmax(outputs, dim=-1) #[3, 499]
            transcription = processor.batch_decode(predicted_ids)
        else:
            ids = torch.argmax(outputs, dim=-1)[0]
            transcription = processor.decode(ids)
    else: #Seamless model
        #audio to text
        output_tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        #output_tokens.keys()
        #['sequences', 'scores', 'encoder_hidden_states', 'decoder_hidden_states', 'past_key_values']
        print(f"Output tokens shape: {output_tokens[0].shape}")#just the 'sequences'
        #torch.Size([1, 27])
        if batchdecode:
            #batch_decode requires a [batch_size, len] as the input, output a list
            transcription = processor.batch_decode(output_tokens[0], skip_special_tokens=True)
        else:
            #output_tokens[0].squeeze() is equivalent to output_tokens[0].tolist()[0]
            ##decode requires vector as the input, string text output
            transcription = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        print(f"Translation from audio: {transcription}")

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
def seamlessm4t_inference(model_name="facebook/seamless-m4t-v2-large", dataset=None, audio_np=None, sr_sampling_rate=16000, device="cuda:0", cache_dir="", target_language='eng'):
    #model_name="facebook/seamless-m4t-v2-large"
    

    
    model = SeamlessM4Tv2Model.from_pretrained(model_name, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast = False)

    model = model.to(device)
    if audio_np is not None:
        inputs = processor(audios=audio_np, sampling_rate=sr_sampling_rate, return_tensors="pt").to(device) 
    else:
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

#https://huggingface.co/facebook/w2v-bert-2.0
def w2vbert_inference(model_name="facebook/w2v-bert-2.0", dataset=None, device="cuda:0", cache_dir=""):
    from transformers import AutoFeatureExtractor, AutoProcessor, Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor
    # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/data/rnd-liu/output/common_voice_0124seq/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    # feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)
    #processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)#, use_fast = False)

    model = Wav2Vec2BertForCTC.from_pretrained(model_name, cache_dir=cache_dir) #vocab_size=len(tokenizer), 

    model = model.to(device)
    sample_rate = dataset[0]["audio"]["sampling_rate"]
    audio = dataset[0]["audio"]["array"]
    text = dataset[0]["sentence"]

    inputs = processor(audio=audio, return_tensors="pt").to(device) #torch.Size([1, 314, 160])
    textids = processor(text=text, return_tensors="pt")

    # input_values = inputs.input_values.to(device)
    # attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)

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

def testmain(mycache_dir, output_dir):
    dataset_name = "mozilla-foundation/common_voice_11_0"
    language = 'en'
    dataset_test = gettestdata(dataset_name, language=language, split="test", sampling_rate=16000, mycache_dir=mycache_dir, streaming=False, samples = 100, task_column="audio", target_column="sentence")
    #raw_datasets = DatasetDict()
    #raw_datasets["train"] = dataset_test

    #"facebook/w2v-bert-2.0" is the basemodel
    w2vbert_inference(model_name="hf-audio/wav2vec2-bert-CV16-en", dataset=dataset_test, device="cuda:0", cache_dir=mycache_dir)

    model_name = "facebook/s2t-wav2vec2-large-en-de"#"facebook/wav2vec2-xls-r-1b-en-to-15"
    wave2vec2seq_inference(model_name, dataset_test, cache_dir=mycache_dir, target_language='de')

    seamlessm4t_inference(model_name="facebook/seamless-m4t-v2-large", dataset=dataset_test, device="cuda:0", cache_dir=mycache_dir, target_language='eng')

    
def Youtube_ASR(model_name, task, device='cuda', filefolder="", language_id='en', mycache_dir='./data', Batch_size = 16, use_video=False, use_tmpfile=True):
    #get the file list in a folder with filename ends with ".json" 
    file_list = [f for f in os.listdir(filefolder) if (f.endswith(".json") and f.startswith(language_id))]
    print(file_list)
    #load the first json file
    infofile = file_list[0] #"./output/video_info.json"
    info_dict =load_json(os.path.join(filefolder,infofile))
    csvpath = info_dict['csvpath']
    df = pd.read_csv(csvpath)
    print(df.head())

    inferencemodel = MyAudioInference(model_name, task=task, device=device, cache_dir=mycache_dir)

    if use_video:
        filepath = os.path.join(filefolder, "clip.mp4")
        clip = mp.VideoFileClip(filepath).audio
    else:
        filepath = os.path.join(filefolder, "audio.mp3")
        clip = mp.AudioFileClip(filepath)
    print("clip duration:", clip.duration)
    audio_array_list = []
    #for loop over each row of df and extract audio from video
    for i, row in df.iterrows():
        audio_obj = {}
        start = min(row['start'], clip.duration)
        end = min(start+ row['duration'], clip.duration)
        print(start, end)
        audio_obj['index'] = i
        audio_obj['start'] = start
        audio_obj['end'] = end
        sub_clip = clip.subclip(start,end)
        #convert audio in sub_clip.audio to numpy array
        if sub_clip.duration < 0.1:
            print(sub_clip.duration)
            print(sub_clip.fps) #44100
            print("Finished the clip")
        fps=sub_clip.fps
        if use_tmpfile == False:
            tt = np.arange(0, sub_clip.duration, 1.0/fps)
            print(tt.max() - tt.min(), tt.min(), tt.max())
            audio_array = sub_clip.to_soundarray(tt=tt, buffersize=fps*15) #https://zulko.github.io/moviepy/_modules/moviepy/audio/AudioClip.html
            print(audio_array.shape) #(455112, 2)
            audio_array = np.transpose(audio_array) #=>(2,n)
            #get numpy array first dimension size
            if audio_array.shape[0]>1 and audio_array.ndim > 1:
                audio_obj['audio_array'] = librosa.to_mono(audio_array)
                print(audio_obj['audio_array'].shape)
        else:
            sub_clip.write_audiofile(os.path.join(filefolder, 'tmp.mp3'))
            #sub_clip.close()
            audio_obj['audio_array'] = librosa.load(os.path.join(filefolder, 'tmp.mp3'), sr=fps)[0]
            print(audio_obj['audio_array'].shape)

        #print(audio_array.dtype)
        #print(audio_array.max())
        #print(audio_array.min())
        #print(audio_array.mean())
        #print(audio_array.std())
        audio_array_list.append(audio_obj)
        if len(audio_array_list)>= Batch_size:
            #run inference
            #for loop over audio_array_list, get each audio_obj, put all the audio_obj['audio_array'] into one new list and run inference
            audio_arrays = [audio_obj['audio_array'] for audio_obj in audio_array_list]
            transcripts = inferencemodel(audio_arrays)
            #write the transcripts to df
            for j, transcript in enumerate(transcripts):
                df_index = audio_array_list[j]['index']
                df.loc[df_index, model_name] = transcript.lower()
            #clear the audio_array_list
            audio_array_list = []
    #save df back to csv file inside of the folder of filefolder
    df.to_csv(csvpath, index=False)
    return df

def test_oneYoutube(model_name, mycache_dir):
    outputfolder = "data/audio"
    clip_url = "https://www.youtube.com/watch?v=7Ood-IE7sx4"
    filepath = os.path.join("data", "clip.mp4")
    #filepath = download_youtube(clip_url, outputfolder=outputfolder)
    clip_paths = clip_video(filepath=filepath, start=0, end=15, step=5, outputfolder=outputfolder)
    print(clip_paths)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inferencemodel = MyAudioInference(model_name, task="audio-asr", target_language='eng', device=device, cache_dir=mycache_dir)
    transcript = inferencemodel(clip_paths, orig_sr=44100)
    print(transcript)    

if __name__ == "__main__":
    

    USE_HPC = False
    USE_Windows = True
    # if USE_HPC:
    #     mycache_dir = deviceenv_set(True, data_path='/data/cmpe249-fa23/Huggingfacecache')
    # elif USE_Windows:
    #     data_path=r"D:\Cache\huggingface"
    #     mycache_dir = deviceenv_set(True, data_path=data_path)
    # else:
    #     mycache_dir = "/DATA10T/Cache"
    
    mycache_dir= r"D:\Cache\huggingface"
    os.environ['HF_HOME'] = mycache_dir
    print("mycache_dir:", mycache_dir)
    model_name = "facebook/seamless-m4t-v2-large"
    test_oneYoutube(model_name, mycache_dir)

    

    model_name = "facebook/wav2vec2-large-robust-ft-libri-960h" #"facebook/wav2vec2-base-960h" #"facebook/wav2vec2-xls-r-300m"
    output_dir = './output'
    #testmain(mycache_dir, output_dir)

    video_id = "Sk1y1auK1xc"
    filefolder = os.path.join("data", "audio", video_id)#"data\audio\Sk1y1auK1xc"
    df = Youtube_ASR(model_name, task="audio-asr", device='cuda', filefolder=filefolder, language_id='en', mycache_dir=mycache_dir, Batch_size = 16)
    print(df.head())

    
    

    


