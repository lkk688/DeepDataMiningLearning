#recreated in 1/22/2014 based on signalAI/hfclassify6_audioasr.py
import evaluate
import math
import torch
from torch.utils.data import DataLoader
import os
from transformers import Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import datetime

from DeepDataMiningLearning.hfaudio.hfutil import valkey, deviceenv_set, get_device
from DeepDataMiningLearning.hfaudio.hfdata import savedict2file, load_audiodataset, getlabels_classifier, vocab_asr, dataset_removecharacters, dataset_castsamplingrate, dataset_preprocessing, filter_datasetlength
from DeepDataMiningLearning.hfaudio.hfmodels import loadmodel, multilingual_tokenizer, load_featureextractor_model, savemodels, load_hfcheckpoint, save_adapterweights, freezemodel, load_featureextractor_seqmodel
from DeepDataMiningLearning.hfaudio.evaluateutil import myEvaluator, evaluate_dataset
from DeepDataMiningLearning.hfaudio.trainutil import get_datacollator, DataCollatorSpeechSeq2SeqWithPadding, get_myoptimizer

def saveargs2file(args, trainoutput):
    args_dict={}
    args_str=' '
    for k, v in vars(args).items():
        args_dict[k]=v
        args_str.join(f'{k}={v}, ')
    print(args_str)
    savedict2file(data_dict=args_dict, filename=os.path.join(trainoutput,'args.json'))

def trainmain(args):
    
    #mycache_dir = deviceenv_set(args.usehpc, args.data_path)
    if os.environ.get('HF_HOME') is not None:
        mycache_dir=os.environ['HF_HOME']
    trainoutput=os.path.join(args.outputdir, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    device, args.useamp = get_device(gpuid=args.gpuid, useamp=args.useamp)
    saveargs2file(args, trainoutput)

    raw_datasets, text_column, target_column, task_column, column_names= load_audiodataset(args.data_name, args.dataconfig, mycache_dir, args.data_type, args.task, args.subtask, args.subset, args.data_path, args.usehpc)

    raw_datasets = dataset_removecharacters(raw_datasets, target_column=target_column)

    labels = None
    id2label = None
    label2id = None
    tokenizer=None
    vocab_path = None
    if args.task == "audio-classification":
        labels, id2label, label2id, num_labels = getlabels_classifier(raw_datasets, target_column=target_column, datatype=args.data_type)
    elif args.task == "audio-asr":
        if args.use_vocabpath:
            vocab_path = trainoutput #"./signalAI/"
            if args.dataconfig != "en":
                a2z_only = False
            else:
                a2z_only = True
            raw_datasets = vocab_asr(raw_datasets, task_column=task_column, target_column=target_column, vocabpath=vocab_path, a2z_only=a2z_only)
            
        #tokenizer = multilingual_tokenizer(args.model_name_or_path, tokenizer_name_or_path=vocab_path, mycache_dir=mycache_dir, output_dir=trainoutput, datasets=raw_datasets, target_column=target_column, target_language=args.target_language, overwrite_lang_vocab=True, overwrite_output_dir=True)
        
        # model, processor, feature_extractor, forward_attention_mask = load_featureextractor_seqmodel(args.model_name_or_path, tokenizer, cache_dir=mycache_dir, outputdir=trainoutput, language=args.target_language, task=args.task, freeze_feature_encoder=args.freeze_feature_encoder, freeze_encoder=args.freeze_basemodel)
        #model, processor, feature_extractor, starting_epoch =load_featureextractor_model(args.model_name_or_path, tokenizer, cache_dir=mycache_dir, config=None, pretrained=args.pretrained, use_adapter=False)
    
    model, feature_extractor, processor, starting_epoch = \
    loadmodel(args.model_name_or_path, custommodel=args.custommodel, \
                task=args.task, id2label=id2label, label2id=label2id, \
                vocab_path=vocab_path, pretrained=args.pretrained, \
                use_adapter = args.use_adapter,
                return_attention_mask=True)

    model = freezemodel(model, unfreezename=args.unfreezename, freezename="", 
                        freeze_feature_encoder=args.freeze_feature_encoder, 
                        freeze_base_model=args.freeze_basemodel, use_adapter =args.use_adapter)
    
    model_input_name = feature_extractor.model_input_names[0]
    print("model_input_name:", model_input_name) #input_values, input_features
    model = model.to(device)
    if args.use_gradientcheckpoint:
        #https://huggingface.co/docs/transformers/main_classes/model
        model.gradient_checkpointing_enable() #save GPU memory 
        #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}) #https://pytorch.org/docs/stable/checkpoint.html
    
    raw_datasets = dataset_castsamplingrate(raw_datasets, sampling_rate=feature_extractor.sampling_rate, audio_column_name=task_column)

    forward_attention_mask = False
    vectorized_datasets = dataset_preprocessing(raw_datasets, tokenizer, processor, task_column=task_column, \
                                                target_column=target_column, max_length_seconds=args.max_length_seconds, \
                                                model_input_name=model_input_name, labels=labels, \
                                                data_type = args.data_type, task=args.task, \
                                                forward_attention_mask=forward_attention_mask)

    #vectorized_datasets = filter_datasetlength(vectorized_datasets, args.min_length_seconds, args.max_length_seconds, sampling_rate=feature_extractor.sampling_rate)

    onesample=next(iter(vectorized_datasets['train']))

    print(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")

    metriceval = myEvaluator(args.task, useHFevaluator=args.hfevaluate, dualevaluator=args.dualevaluate,\
                              labels=labels, processor=processor, mycache_dir=mycache_dir, output_path=trainoutput)

    data_collator = get_datacollator(processor, args.task, padding=True)
    # data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    #     processor=processor,
    #     decoder_start_token_id=model.config.decoder_start_token_id,
    #     forward_attention_mask=forward_attention_mask,
    # )

    #['HFTrainer','CustomTrain', 'NoTrain']
    if args.trainmode == 'HFTrainer':
        training_args = Seq2SeqTrainingArguments(
            trainoutput,
            group_by_length=True,#makes training more efficient by grouping training samples of similar input length into one batch.
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learningrate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.total_epochs,
            #warmup_ratio=args.warmup_ratio, #0.1,
            warmup_steps=args.warmup_steps, #500,
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            #metric_for_best_model=myEvaluator.#"accuracy",
            fp16=args.use_fp16,
            push_to_hub=False,
            #gradient_checkpointing=True,#reduce GPU memory, or use model.gradient_checkpointing_enable()
        )
        # Initialize our trainer
        trainer = Seq2SeqTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=vectorized_datasets["train"],
            eval_dataset=vectorized_datasets[valkey],
            compute_metrics=metriceval.compute_metrics,
            tokenizer=processor.feature_extractor, #feature_extractor,
        )

        checkpoint = load_hfcheckpoint(args.checkpointfolder)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        #Perform evaluation
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        #save_adapterweights(model, args.target_language, trainoutput)

    elif args.trainmode == 'CustomTrain':
        
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.batch_size,
        )
        eval_dataloader = DataLoader(
            vectorized_datasets[valkey], collate_fn=data_collator, batch_size=args.batch_size
        )

        results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)

        # Optimizer
        optimizer = get_myoptimizer(model, learning_rate=args.learningrate, weight_decay=args.weight_decay)
        
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = (args.total_epochs - starting_epoch) * num_update_steps_per_epoch

        num_warmup_steps = args.warmup_steps #10
        lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type, #["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_train_steps,
            )
        
        #recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps /  num_update_steps_per_epoch) + starting_epoch
        total_batch_size = args.batch_size * args.gradient_accumulation_steps

        progress_bar = tqdm(range(max_train_steps))
        completed_steps = 0
        #starting_epoch = 0
        if args.useamp == True:
            # Creates a GradScaler once at the beginning of training.
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(starting_epoch, num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                # Enables autocasting for the forward pass (model + loss)
                if args.useamp == True:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**batch)
                        loss = outputs.loss
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                #backward
                if args.useamp == True:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if args.useamp == True:
                            scaler.step(optimizer)
                            # Updates the scale for next iteration.
                            scaler.update()
                        else:
                            optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1
            #Evaluation
            results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)   
            print(f"epoch {epoch}, loss: {loss}, evaluation: {metriceval.metricname}")
            print("Evaluation result:", results)
            # Save the results
            savedict2file(data_dict=results, filename=os.path.join(trainoutput, f"epoch{epoch}_"+"eval_results.json"))
            
            savemodels(model, optimizer, epoch, trainoutput)
            #feature_extractor.save_pretrained(trainoutput)
    else: #NoTrain
        eval_dataloader = DataLoader(
            raw_datasets[valkey], collate_fn=None, batch_size=args.batch_size
        )
        results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)
        print("No training, evauate only:", results)

    # using now() to get current time
    current_time = datetime.datetime.now()
    # Printing value of now.
    print("Time now is:", current_time)
    print("Finished")
    
    return ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #data related arguments
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="common_voice",
                    help='data name: common_voice, librispeech_asr, aesdd(local path), timit, common_language, superb, google/fleurs, minds14, marsyas/gtzan')
    parser.add_argument('--dataconfig', type=str, default='zh-CN',
                    help='dataset_config_name, e.g., common_voice subset en, zh-CN')
    parser.add_argument('--target_language', type=str, default='en',
                    help='target_language: en')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache" "/DATA10T/Cache"
    #model related arguments
    parser.add_argument('--model_name_or_path', type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
                    help='Model checkpoint name from HF, ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt, facebook/w2v-bert-2.0, hf-audio/wav2vec2-bert-CV16-en, facebook/w2v-bert-2.0, jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn, facebook/mms-1b-all, jonatasgrosman/wav2vec2-large-xlsr-53-english, TencentGameMate/chinese-wav2vec2-base, facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-large-xlsr-53, anton-l/xtreme_s_xlsr_300m_minds14, facebook/wav2vec2-base-960h, "facebook/wav2vec2-base", ntu-spml/distilhubert')
    #autotokenizer
    parser.add_argument('--autotokenizer', default=True, action='store_true',
                    help='If some models contains the tokenizer, autotokenizer=True means use the default buildin tokenizer')
    parser.add_argument('--use_vocabpath', default=False, action='store_true', help='Use new vocab file')
    parser.add_argument('--use_adapter', default=False, action='store_true', help='Add adapter')
    parser.add_argument('--checkpointfolder', type=str, default="",
                    help='Model training checkpoint to resume')
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--custommodel', default=False, action='store_true', help='Change model') 
    parser.add_argument('--task', type=str, default="audio-asr",
                    help='tasks: audio-classification, openqa, translation, summarization, QA')
    parser.add_argument('--subtask', type=str, default="intent-classification",
                    help='Sub tasks')
    parser.add_argument('--hfevaluate', default=True, action='store_true',
                    help='perform evaluation via HFevaluate or localevaluate')
    parser.add_argument('--dualevaluate', default=False, action='store_true',
                    help='perform evaluation via HFevaluate and localevaluate')
    # parser.add_argument('--unfreezename', type=str, default="",
    #                 help='Unfreezename in models') #adapter
    parser.add_argument('--unfreezename', nargs='+', help='Unfreezename in models, could be list', required=False)
    parser.add_argument('--freeze_feature_encoder', default=False, action='store_true', help='Freeze the featureencoder')
    parser.add_argument('--freeze_basemodel', default=False, action='store_true', help='Freeze the basemodel')
    #training related arguments
    parser.add_argument('--outputdir', type=str, default="/data/rnd-liu/output/", help='output path') #r"E:\output" "./output" "/DATA10T/output/"
    parser.add_argument('--traintag', type=str, default="0301w2vzh_nonewvocal",
                    help='Name the current training')
    # parser.add_argument('--training', default=True, action='store_true',
    #                 help='Perform training')
    parser.add_argument('--trainmode', default="CustomTrain", choices=['HFTrainer','CustomTrain', 'NoTrain'], help='Training mode')
    #vocab_path
    parser.add_argument('--use_fp16', default=False, action='store_true',
                    help='Use HPC')
    parser.add_argument('--use_gradientcheckpoint', default=False, action='store_true',
                    help='Use gradientcheckpoint')#gradient_checkpointing_enable
    parser.add_argument('--usehpc', default=True, action='store_true',
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', default=False, action='store_true',
                    help='Use Huggingface accelerator')
    parser.add_argument('--useamp', default=True, action='store_true',
                    help='Use pytorch amp in training')
    parser.add_argument('--gpuid', default=3, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=20, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=3e-4, type=float, help='Learning rate')
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    #warmup_ratio
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps in learning rate scheduling.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass, ref: https://kozodoi.me/blog/20210219/gradient-accumulation.",
    )
    parser.add_argument(
        "--max_length_seconds",
        type=int,
        default=8, #20,
        help=(
            "Audio clips will be randomly cut to this length during training if the value is set.."
        ),
    )
    parser.add_argument(
        "--min_length_seconds",
        type=int,
        default=1, #20,
        help=(
            "Audio clips will be randomly cut to this length during training if the value is set.."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help=(
            "Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=384, #128, #1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    trainmain(args)