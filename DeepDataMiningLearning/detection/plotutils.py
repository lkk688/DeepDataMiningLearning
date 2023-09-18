#ref: https://github.com/lkk688/MultiModalDetector/blob/master/Myutils/plotresults.py
import cv2
#from utils.plotresults import show_image_bbxyxy
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
#%matplotlib inline
from PIL import Image 
import matplotlib.pyplot as plt

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

def show_image_bbxyxy(image, pred_bbox, pred_ids, title, INSTANCE_CATEGORY_NAMES, savefigname=None):
    """Show a camera image and the given camera labels."""
        
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
            xy=patch[0],#(patch[0], patch[1]), #xmin ymin
            width=patch[1][0]-patch[0][0],#patch[2] - patch[0],
            height=patch[1][1]-patch[0][1],#patch[3] - patch[1],
            linewidth=4,
            edgecolor=colorlabelnormalized,#"red",
            facecolor='none'))
            ax.text(patch[0][0], patch[0][1], label, color=colorlabelnormalized, fontsize=12)
            #ax.text(patch[0][0], patch[0][1], label, bbox=dict(facecolor='red', alpha=0.5))#fontsize=8)
        
    ax.imshow(image)
    
    ax.title.set_text(title)
    ax.grid(False)
    ax.axis('off')
    
    if savefigname is not None:
        fig.savefig(savefigname)
    
    #fig.savefig(f"output/test_frame_{i}.png", dpi=fig.dpi)
#     plt.show()

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