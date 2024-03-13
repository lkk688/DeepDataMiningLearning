#ref: https://github.com/lkk688/MultiModalDetector/blob/master/Myutils/plotresults.py
import cv2
#from utils.plotresults import show_image_bbxyxy
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
#%matplotlib inline
from PIL import Image 
import matplotlib.pyplot as plt
import torch
import numpy as np

INSTANCE_Color = {
    'Unknown':'black', 'Vehicles':'red', 'Pedestrians':'green', 'Cyclists':'purple'
}#'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'



def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    #color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    color = [int((p * ((label+5*label) ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES):
    numclasses=len(INSTANCE_CATEGORY_NAMES)
    pred_labels=[]
    for pred_i in list(pred_ids):
        if pred_i >= numclasses:
            pred_labels.append('Unknown')
        else:
            pred_labels.append(INSTANCE_CATEGORY_NAMES[pred_i])
    return pred_labels

def matplotlibshow_image_bbxyxy(image, pred_bbox, pred_ids, title, INSTANCE_CATEGORY_NAMES, savefigname=None):
    """Show a camera image (HWC format) and the given camera labels."""
        
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    boxnum=len(pred_bbox)
    #print(boxnum)
    if len(pred_ids)<1:
        print("No object detected")
        return image
    else:
        #pred_labels = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred_ids) ]
        pred_labels = convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES)
        #print(pred_labels)
        for i in range(boxnum):#patch in pred_bbox:
            patch=pred_bbox[i]
            #print(patch)
            colorlabel=compute_color_for_labels(pred_ids[i]) #INSTANCE_Color[label]
            #print(colorlabel)#RGB value 0-255
            colorlabelnormalized = [float(i)/255 for i in colorlabel] #0-1
            label=pred_labels[i]
            #print(label)
            ax.add_patch(Rectangle(
            xy=(patch[0], patch[1]), #xmin ymin
            width=patch[2] - patch[0],
            height=patch[3] - patch[1],
            linewidth=4,
            edgecolor=colorlabelnormalized,#"red",
            facecolor='none'))
            ax.text(patch[0], patch[1], label, color=colorlabelnormalized, fontsize=15)
            #ax.text(patch[0][0], patch[0][1], label, bbox=dict(facecolor='red', alpha=0.5))#fontsize=8)
        
    ax.imshow(image)
    
    ax.title.set_text(title)
    ax.grid(False)
    ax.axis('off')
    
    if savefigname is not None:
        fig.savefig(savefigname)
    
    #fig.savefig(f"output/test_frame_{i}.png", dpi=fig.dpi)
#     plt.show()

from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

def pil_tonp(img_pil, outputformat='CHW'):
    #convert PIL image to numpy
    a = np.asarray(img_pil)
    print("input image shape", a.shape) #HWC format (height, width, color channels)
    if outputformat=='CHW':
        imgdata = a.transpose((2, 0, 1)) #CHW (color channels, height, width)
    elif outputformat=='HWC':
        imgdata = a
    return imgdata

import cv2
def npimage_RGBchange(imgdata, fromformat='BGR', toformat='RGB'): 
    #imgdata is HWC format
    im = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    return im

##torch read_image get format CHW (color channels, height, width)
#pred_bbox_tensor in (xmin, ymin, xmax, ymax) format
def drawbbox_topil(imgdata_np, pred_bbox_np, labels_str, colors='red'):
    #imgdata_np CHW format
    pred_bbox_tensor = torch.from_numpy(pred_bbox_np)
    box = draw_bounding_boxes(torch.from_numpy(imgdata_np), boxes=pred_bbox_tensor,
                            labels=labels_str,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    return im

def draw_boxes(image, pred_bbox, pred_ids, pred_score, INSTANCE_CATEGORY_NAMES):
    boxnum=len(pred_bbox)
    #print(boxnum)
    if len(pred_ids)<1:
        print("No object detected")
        return image
    else:
        #pred_labels = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred_ids) ]
        pred_labels = convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES)
        #print(pred_labels)
        pred_score_str=["%.2f" % i for i in pred_score]
        for i in range(boxnum):#patch in pred_bbox:
            patch=pred_bbox[i] # [ (xmin, ymin), (xmax, ymax)]
            #patch[0] (xmin, ymin)
            #patch[1] (xmax, ymax)
            x1=int(patch[0][0])
            y1=int(patch[0][1])
            x2=int(patch[1][0])
            y2=int(patch[1][1]) #cv2.rectangle need int input not float
            colorlabel=compute_color_for_labels(pred_ids[i]) #RGB value 0-255
            label=pred_labels[i]+" "+pred_score_str[i]
            labelscale=1
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, labelscale , 2)[0] #font scale: 1, font_thickness: 1
            cv2.rectangle(image, (x1, y1), (x2,y2), colorlabel, 2)
            cv2.rectangle(image, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), colorlabel, -1) #-1 is fill the rectangle
            cv2.putText(image,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, labelscale, [255,255,255], 2)
    return image

def draw_trackingboxes(image, pred_bbox, identities=None, track_class=None, INSTANCE_CATEGORY_NAMES=None):
    boxnum=len(pred_bbox)
    #print(boxnum)
    if boxnum<1:
        print("No object detected")
        return image
    else:
        if track_class is not None:
            pred_labels = convertIDtolabel(track_class, INSTANCE_CATEGORY_NAMES)
        for i in range(boxnum):#patch in pred_bbox:
            patch=pred_bbox[i] # [ xmin, ymin, xmax, ymax]
            #patch[0] (xmin, ymin)
            #patch[1] (xmax, ymax)
            x1=int(patch[0])
            y1=int(patch[1])
            x2=int(patch[2])
            y2=int(patch[3]) #cv2.rectangle need int input not float
            id = int(identities[i]) if identities is not None else 0    

            
            if track_class is not None:
                label=pred_labels[i]+" T:"+str(id)
                colorlabel=compute_color_for_labels(track_class[i]) #RGB value 0-255
            else:
                label="T:"+str(id)
                colorlabel = compute_color_for_labels(id)
            labelscale=1
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, labelscale , 2)[0] #font scale: 1, font_thickness: 1
            cv2.rectangle(image, (x1, y1), (x2,y2), colorlabel, 2)
            cv2.rectangle(image, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), colorlabel, -1) #-1 is fill the rectangle
            cv2.putText(image,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, labelscale, [255,255,255], 2)
    return image



def show_imagewithscore_bbxyxy(image, pred_bbox, pred_ids, pred_score, title, INSTANCE_CATEGORY_NAMES, savefigname=None):
    """Show a camera image and the given camera labels."""
        
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    boxnum=len(pred_bbox)
    #print(boxnum)
    #pred_ids may contain 80, but INSTANCE_CATEGORY_NAMES only has 79
    #pred_labels = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred_ids)]
    pred_labels = convertIDtolabel(pred_ids, INSTANCE_CATEGORY_NAMES)

    pred_score_str=["%.2f" % i for i in pred_score]
    #print(pred_labels)
    for i in range(boxnum):#patch in pred_bbox:
        patch=pred_bbox[i]
        #print(patch)
        colorlabel=compute_color_for_labels(pred_ids[i]) #INSTANCE_Color[label]
        #print(colorlabel)#RGB value 0-255
        colorlabelnormalized = [float(i)/255 for i in colorlabel] #0-1
        label=pred_labels[i]+" "+pred_score_str[i]
        #print(label)
        ax.add_patch(Rectangle(
        xy=patch[0],#(patch[0], patch[1]), #xmin ymin
        width=patch[1][0]-patch[0][0],#patch[2] - patch[0],
        height=patch[1][1]-patch[0][1],#patch[3] - patch[1],
        linewidth=3,
        edgecolor=colorlabelnormalized,#"red",
        facecolor='none'))
        #ax.text(patch[0][0], patch[0][1], label, color=colorlabelnormalized, fontsize=14)
        ax.text(patch[0][0], patch[0][1], label, bbox=dict(facecolor=colorlabelnormalized, alpha=0.4), fontsize=14)#fontsize=8)
        
        
    ax.imshow(image)
    
    ax.title.set_text(title)
    ax.grid(False)
    ax.axis('off')
    
    if savefigname is not None:
        fig.savefig(savefigname)
    
    #fig.savefig(f"output/test_frame_{i}.png", dpi=fig.dpi)
#     plt.show()

#add from https://github.com/lkk688/myyolov7/blob/main/utils/plots.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)

#added for hfvisionmain
def pixel_values2img(pixel_values):
    img_np=pixel_values.cpu().squeeze(dim=0).permute(1, 2, 0).numpy() #CHW->HWC 
    print(img_np.shape) # (800, 1066, 3)
    img_np = (img_np-np.min(img_np))/(np.max(img_np)-np.min(img_np)) 
    img_np=(img_np * 255).astype(np.uint8)
    image = Image.fromarray(img_np, 'RGB') #pil image
    return image

def draw_objectdetection_predboxes(image, pred_boxes, scores, labels, id2label, threshold=0.1, save_path="output/Imagepredplot.png"):
    
    # pred_boxes = box_convert(pred_boxes, 'xywh', 'xyxy')
    # pred_boxes = pred_boxes.numpy()
    #print(pred_boxes) #[100,4]

    filterscore = scores > threshold
    filterlabel = labels >0
    selectedindex =  filterscore & filterlabel
    pred_boxes = pred_boxes[selectedindex]
    scores = scores[selectedindex]
    labels = labels[selectedindex]
    #scores = scores.flatten()#[1, 100] to [100]
    #labels = labels.flatten()
    #boxes = center_to_corners_format(out_bbox)
    #(center_x, center_y, width, height) =>(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    #results.append({"scores": score, "labels": label, "boxes": box})
    
    width, height = image.size #1066, 800
    draw = ImageDraw.Draw(image, "RGBA")
    box_len = len(pred_boxes)
    for index in range(box_len):
        pred = pred_boxes[index]
        score= scores[index]
        if score < threshold:
            continue
        xc, yc, w, h = tuple(pred) #(center_x, center_y, width, height) normalized
        x, y, x2, y2 = (xc-w/2)*width, (yc-h/2)*height, (xc+w/2)*width, (yc+h/2)*height
        draw.rectangle((x, y, x2, y2), outline="red", width=1) #[xmin, ymin, xmax, ymax]
        # box = annotation['bbox']
        # class_idx = annotation['category_id']
        # x,y,w,h = tuple(box)
        # draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        class_idx = labels[index]
        draw.text((x, y), id2label[class_idx], fill='white')
    if save_path:
        image.save(save_path)

def draw_objectdetection_results(image, results, id2label, save_path="output/Imageresultplot.png"): #model.config.id2label
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()] #[471.16, 209.09, 536.17, 347.85]#[xmin, ymin, xmax, ymax]
        print(
            f"Detected {id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1) #[xmin, ymin, xmax, ymax]
        draw.text((x, y), id2label[label.item()], fill="white")
    if save_path:
        image.save(save_path)


def draw_annobox2image(image, annotations, id2label=None, save_path="output/ImageDrawcoco.png"):
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    for annotation in annotations:
        x, y, w, h = tuple(annotation) #[xmin, ymin, xmax, ymax]
        x, y, x2, y2 = x*width, y*height, (x+w)*width, (y+h)*height
        draw.rectangle((x, y, x2, y2), outline="red", width=1) #[xmin, ymin, xmax, ymax]
        # box = annotation['bbox']
        # class_idx = annotation['category_id']
        # x,y,w,h = tuple(box)
        # draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        # if id2label is not None:
        #     draw.text((x, y), id2label[class_idx], fill='white')
    if save_path:
        image.save(save_path)
    return image

#https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation
#"coco": [x_min, y_min, width, height] in pixels 
#pascal_voc: [x_min, y_min, x_max, y_max] in pixels
#albumentations  [x_min, y_min, x_max, y_max] normalized
#yolo: [x_center, y_center, width, height] normalized
#torchvision 'xyxy' box_convert ['xyxy', 'xywh', 'cxcywh']
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.ops import box_convert
def draw2pil(image, bbox, category, categories, bbox_format='coco', filepath=None):
    if not torch.is_tensor(image):
        image = pil_to_tensor(image)
    if not torch.is_tensor(bbox):
        boxes_in = torch.tensor(bbox) #annotations['bbox']
    else:
        boxes_in = bbox
    if bbox_format=='coco': #'xywh':
        boxes_new = box_convert(boxes_in, 'xywh', 'xyxy') #['xyxy', 'xywh', 'cxcywh']
    elif bbox_format=='pascal_voc':
        boxes_new = boxes_in
    labels = [categories.int2str(x) for x in category] #annotations['category']
    image_annoted=to_pil_image(
        draw_bounding_boxes(
            image,
            boxes_new, #Boxes need to be in (xmin, ymin, xmax, ymax)
            colors="red",
            labels=labels,
        )
    )
    if filepath:
        image_annoted.save(filepath)
    return image_annoted