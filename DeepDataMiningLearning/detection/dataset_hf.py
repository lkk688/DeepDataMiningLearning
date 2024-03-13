import torch
import torchvision
import albumentations#pip install albumentations
import os
import requests
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import numpy as np
import json
import evaluate
#from DeepDataMiningLearning.hfvision_inference import save_coco_annotation_file_images
from tqdm.contrib import tzip

# format annotations the same as for training, no need for data augmentation
def val_formatted_anns(image_id, objects, format='coco'):
    annotations = []
    boxlen = len(objects["bbox"])
    if 'id' in objects.keys():
        use_id = True
    else:
        use_id = False
    for i in range(0, boxlen):
        bbox = objects["bbox"][i]
        if format!='coco' and format=='pascal_voc':
            xmin, ymin, xmax, ymax = bbox
            width = xmax-xmin
            height = ymax-ymin
            bbox = [xmin, ymin, width, height]
        new_ann = {
            "id": objects["id"][i] if use_id else i,
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": bbox,
        }
        annotations.append(new_ann)

    return annotations

def check_boxsize(bbox, height=None, width=None, format='coco'):
    errobox=False
    newbbox=[]
    for box in bbox:
        if format=='coco':
            xmin, ymin, w, h = box
        else:
            xmin, ymin, xmax, ymax = box
            w = max(xmax-xmin, 1)
            h = max(ymax-ymin, 1)
        if xmin+w>width:
            w = max(width-xmin, 1)
            errobox = True
        if ymin+h>height:
            h = max(height - ymin, 1)
            errobox = True
        if errobox:
            box=[xmin, ymin, w, h]
        if xmin > width or ymin > height:
            box=[0, 0, 1, 1]
        if format=='coco':
            newbbox.append(box)
        else:
            newbbox.append([xmin, ymin, xmin+w, ymin+h])
    return newbbox, errobox

def save_coco_annotation_file_images(dataset, id2label, path_output, path_anno, format='coco', json_only=False):
    output_json = {}

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if path_anno is None:
        path_anno = os.path.join(path_output, "coco_anno.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in tqdm(dataset):
        ann = val_formatted_anns(example["image_id"], example["objects"], format)#list of dicts
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    if json_only!=True:
        #for im, img_id in zip(dataset["image"], dataset["image_id"]):
        for im, img_id in tzip(dataset["image"], dataset["image_id"]):
            path_img = os.path.join(path_output, f"{img_id}.png")
            im.save(path_img)

    return path_output, path_anno

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx) #img: (400, 300), target: list[dicts] each dict has one box 

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target} #target (top_left_x, top_left_y, width, height)
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension [3, 800, 1066]
        target = encoding["labels"][0]  # remove batch dimension, dict

        return {"pixel_values": pixel_values, "labels": target}#, "img":img, "image_id": image_id}#adding "img" and "image_id"
    
class HFCOCODataset(torch.utils.data.Dataset):
    def __init__(self, dataset, id2label, dataset_folder, coco_anno_json=None, data_type='huggingface', format='coco', image_processor=None):
        #Convert HF dataset to COCO format
        if coco_anno_json is None:
            coco_anno_json = os.path.join(dataset_folder, 'coco_anno.json')
        if data_type == "huggingface" and not os.path.exists(coco_anno_json):
            print("Convert HF dataset to COCO format into folder:", dataset_folder)
            dataset_folder, coco_anno_json = save_coco_annotation_file_images(dataset, id2label=id2label, path_output=dataset_folder, path_anno=coco_anno_json, format=format, json_only=False)
        self.dataset_folder = dataset_folder
        #create CocoDetection dataset
        self.ds_coco = CocoDetection(dataset_folder, image_processor, coco_anno_json)
        #{"pixel_values": pixel_values, "labels": target} after processor
        print(len(self.ds_coco))
        self.image_ids = self.ds_coco.coco.getImgIds()
        print(len(self.image_ids))
        self.coco = self.ds_coco.coco #pycocotools.coco.COCO object
        cats = self.coco.cats #dict of dict
        self.id2label = {k: v['name'] for k,v in cats.items()}

    def get_img(self, image_id):
        images = self.coco.loadImgs(image_id)
        image = images[0] #dict with image info, 'file_name'
        image = Image.open(os.path.join(self.dataset_folder, image['file_name']))
        return image
    
    def get_anno(self, image_id):
        annotations = self.coco.imgToAnns[image_id]#list of dicts
        return annotations
    
    def draw_anno2image(self, image, annotations, id2label=None, save_path="output/ImageDrawcoco.png"):
        draw = ImageDraw.Draw(image, "RGBA")
        for annotation in annotations:
            box = annotation['bbox']
            class_idx = annotation['category_id']
            x,y,w,h = tuple(box)
            draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
            if id2label is not None:
                draw.text((x, y), id2label[class_idx], fill='white')
        if save_path:
            image.save(save_path)
        return image

    def test_cocodataset(self, index=None):
        test_coco= self.ds_coco[0] #pixel_values[3, 693, 1333] labels dict 
        # let's pick a random image
        if index is None:
            index = np.random.randint(0, len(self.image_ids))
        image_id = self.image_ids[index] #
        print('Image n°{}'.format(image_id))
        image = self.get_img(image_id)
        annotations = self.get_anno(image_id)
        iamge_draw = self.draw_anno2image(image, annotations, self.id2label)
        
    def __getitem__(self, index):
        datadict = self.ds_coco[index] #img, target
        return datadict

    def __len__(self):
        return len(self.ds_coco)

#results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)
def test_evaluate_dataset(model, dataset, id2label, dataset_folder, coco_anno_json, data_type, format, device, image_processor, collate_fn):
    #id2label = model.id2label
    if coco_anno_json is None:
        coco_anno_json = os.path.join(dataset_folder, 'coco_anno.json')
    if data_type == "huggingface" and not os.path.exists(coco_anno_json):
        dataset_folder, coco_anno_json = save_coco_annotation_file_images(dataset, id2label=id2label, path_output=dataset_folder, path_anno=coco_anno_json, format=format, json_only=False)
    # path_output = 'output/coco/'
    # path_anno = 'output/coco/cppe5_ann.json'
    test_ds_coco_format = CocoDetection(dataset_folder, image_processor, coco_anno_json)
    print(len(test_ds_coco_format))
    image_ids = test_ds_coco_format.coco.getImgIds()
    test_coco= test_ds_coco_format[0] #pixel_values[3, 693, 1333] labels dict 
    # let's pick a random image
    image_id = image_ids[15] #np.random.randint(0, len(image_ids))
    print('Image n°{}'.format(image_id))
    image = test_ds_coco_format.coco.loadImgs(image_id)[0] #dict with image info, 'file_name'
    image = Image.open(os.path.join(dataset_folder, image['file_name']))
    annotations = test_ds_coco_format.coco.imgToAnns[image_id]#list of dicts
    draw = ImageDraw.Draw(image, "RGBA")
    cats = test_ds_coco_format.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='white')
    image.save("output/ImageDrawcoco.png")

    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
    test_data = next(iter(val_dataloader))
    print(test_data.keys()) #['pixel_values', 'pixel_mask', 'labels'] 'labels' is list of dicts
    print(test_data["pixel_values"].shape) #[8, 3, 840, 1333]
    print(test_data["pixel_mask"].shape) #[8, 840, 1333]

    model = model.eval().to(device)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"].to(device)#[8, 3, 840, 1333]
            pixel_mask = batch["pixel_mask"].to(device)#[8, 840, 1333]

            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized, list of dicts

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask) #DetrObjectDetectionOutput

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0) #[8,2] shape
            results = image_processor.post_process_object_detection(outputs,  threshold=0.0, target_sizes=orig_target_sizes)  # convert outputs of model to COCO api, list of dicts
            module.add(prediction=results, reference=labels)
            del batch

    results = module.compute() #iou_bbox key
    print(results)


# if __name__ == "__main__":
#     #test HFCOCODataset
#     hfcoco = HFCOCODataset(dataset, id2label, dataset_folder, coco_anno_json=None, data_type='huggingface', format='coco', image_processor=None)