#import DeepDataMiningLearning.vision.myevaluate as myevaluate
import numpy as np
from tqdm.auto import tqdm
from DeepDataMiningLearning.detection.plotutils import draw2pil, pixel_values2img, draw_objectdetection_predboxes, draw_objectdetection_results
import torch
from torch import nn


class myEvaluator:
    def __init__(self, task, useHFevaluator=False, dualevaluator=False, processor=None, coco=None, mycache_dir=None):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = task
        self.preds = []
        self.refs = []
        #self.labels = labels
        self.processor = processor
        self.HFmetric = None
        if self.task == "image-classification":
            self.metricname = "accuracy" #"mse" "wer"
        elif self.task == "object-detection":
            self.metricname = "coco"
            #prepare
        else:
            self.metricname = "accuracy"
        self.LOmetric = None
        if self.useHFevaluator and self.task=="object-detection":
            self.HFmetric = myevaluate.load("ybelkada/cocoevaluate", coco=coco) #test_ds_coco_format.coco)
        elif self.useHFevaluator:
            # Load the accuracy metric from the datasets package
            self.HFmetric = myevaluate.load(self.metricname) #evaluate.load("mse")
            

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    #eval_pred is EvalPrediction type
    def compute_metrics(self, eval_pred): #: EvalPrediction):
        #preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions #(1000, 593, 46)
        preds, labels = eval_pred
        if self.metricname == "accuracy":
            """Computes accuracy on a batch of predictions"""
            preds = np.argmax(preds, axis=1)
            #return self.HFmetric.compute(predictions=predictions, references=labels)
        elif self.metricname == "mse":
            preds = np.squeeze(preds)
            #return self.HFmetric.compute(predictions=preds, references=label_ids)
        return self.compute(predictions=preds, references=labels)

    def mycompute(self, predictions=None, references=None):
        predictions = np.array(predictions)
        references = np.array(references)
        if self.metricname == "accuracy":
            eval_result = (predictions == references).astype(np.float32).mean().item()
            # if self.labels:
            #     print("Classification report", classification_report(references, predictions, target_names=self.labels))
        elif self.metricname == "mse": #mse
            eval_result = ((predictions - references) ** 2).mean().item()
        else:
            eval_result = None
        results = {self.metricname: eval_result}
        return results
    
    def compute(self, predictions=None, references=None):
        results = {}
        if predictions is not None and references is not None:
            if self.useHFevaluator:
                results = self.HFmetric.compute(predictions=predictions, references=references)
            else: 
                results = self.mycompute(predictions=predictions, references=references)
            #print("HF evaluator:", results)
            if not isinstance(results, dict):
                #output is float, convert to dict
                results = {self.metricname: results}
        else: #evaluate the whole dataset
            if self.useHFevaluator:
                results = self.HFmetric.compute()
                print("HF evaluator result1:", results)#iou_bbox 
                #results2 = self.HFmetric.compute(predictions=self.preds, references=self.refs) #the same results
                #print("HF evaluator result2:", results2)
                if not isinstance(results, dict):
                    #wer output is float, convert to dict
                    results = {self.metricname: results}
            else:
                results = self.mycompute(predictions=self.preds, references=self.refs)
            self.preds.clear()
            self.refs.clear()
        if self.task == "object-detection":
            results = results['iou_bbox']
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator == True:
            if self.task=="object-detection":
                self.HFmetric.add(prediction=predictions, reference=references)
            else:
                self.HFmetric.add_batch(predictions=predictions, references=references)
        else:
            #self.preds.append(predictions)
            self.refs.extend(references)
            self.preds.extend(predictions)
        #references: list of list
        # for ref in references:
        #     self.refs.append(ref[0])
        #print(len(self.refs))
        

def evaluate_dataset(model, val_dataloader, task, metriceval, device, image_processor=None, accelerator=None):
    
    model = model.eval().to(device)
    for step, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"].to(device)#[8, 3, 840, 1333]
        if "pixel_mask" in batch:
            pixel_mask = batch["pixel_mask"].to(device)#[8, 840, 1333]
        else:
            pixel_mask = None
        #batch = {k: v.to(device) for k, v in batch.items()}
        #"pixel_values" [8, 3, 840, 1333]
        #"pixel_mask" [8, 840, 1333]
        # "labels" 
        with torch.no_grad():
            #outputs = model(**batch)
            if pixel_mask is None:
                outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask) #DetrObjectDetectionOutput
        
        if task == "image-classification":
            predictions = outputs.logits.argmax(dim=-1)
            if accelerator is not None:
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            else:
                predictions, references = predictions, batch["labels"]
        elif task == "object-detection":
            references = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  #resized + normalized, list of dicts ['size', 'image_id', 'boxes'[8,4], 'class_labels', 'area', 'orig_size'[size[2]]]
            orig_target_sizes = torch.stack([target["orig_size"] for target in references], dim=0) #[8,2] shape
            # convert outputs of model to COCO api, list of dicts
            predictions = image_processor.post_process_object_detection(outputs,  threshold=0.0, target_sizes=orig_target_sizes) 
            #list of dicts ['scores', 'labels'[100], 'boxes'(100,4)]
            
            id2label = model.config.id2label 
            #print(batch["labels"][0].keys()) #['size', 'image_id', 'class_labels', 'boxes', 'area', 'iscrowd', 'orig_size']
            image = pixel_values2img(pixel_values)
            pred_boxes = outputs['pred_boxes'].cpu().squeeze(dim=0).numpy() #(100,4) normalized (center_x, center_y, width, height)
            prob = nn.functional.softmax(outputs['logits'], -1) #[1, 100, 92]
            scores, labels = prob[..., :-1].max(-1) #[1, 100] [1, 100]
            scores = scores.cpu().squeeze(dim=0).numpy() #(100,)
            labels = labels.cpu().squeeze(dim=0).numpy() #(100,)
            draw_objectdetection_predboxes(image.copy(), pred_boxes, scores, labels, id2label) #DetrObjectDetectionOutput
            #print(batch["labels"])#list of dicts
            
            #the image size is the not correct
            draw_objectdetection_results(image, predictions[0], id2label)
        metriceval.add_batch(
            predictions=predictions,
            references=references,
        )
        del batch

    eval_metric = metriceval.compute()#metric.compute()
    #print(eval_metric)
    # Printing key-value pairs as tuples
    print("Eval metric Key-Value Pairs:", list(eval_metric.items()))
    return eval_metric