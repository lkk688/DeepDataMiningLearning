#from https://github.com/lkk688/yolov5/blob/master/data/cocojsontoyolo.py
#ref: https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
import json

import cv2
import pandas as pd
from PIL import Image

import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import ExifTags
from tqdm import tqdm

def make_dirs(dir, deleteold=True):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        if deleteold == True:
            shutil.rmtree(dir)  # delete dir
        else:
            print("Folders already exist!")
            return dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x

def convert_waymococo_json(source_dir, json_file, save_dir='./waymococoyolo', deleteold=True, copyimage=True):
    save_dir = make_dirs(save_dir, deleteold)  # create output directory with labels and images folder

    source_folder=Path(source_dir) #source images folder

    # Import json file
    fn = Path(save_dir) / 'labels' #/ json_file.stem.replace('instances_', '')  # folder name
    #fn.mkdir()
    with open(json_file) as f:
        data = json.load(f)
    
    # Create image dict
    #images = {'%g' % x['id']: x for x in data['images']}
    images = {'%g' % int(x['id']): x for x in data['images']}#our converted COCO 'id' field is str not int

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['iscrowd']:
            continue

        #img = images['%g' % x['image_id']]
        imageidstr=x['image_id']
        img = images['%g' % int(imageidstr)]
        h, w, f = img['height'], img['width'], img['file_name']

        #New: copy image files, img['id']
        #img['file_name']='images/10203656353524179475_7625_000_7645_000/ change to img['image_id']
        if copyimage == True:
            newimagefilename=imageidstr+'.jpg'
            destinationfile= save_dir / 'images' / newimagefilename
            if not destinationfile.is_file():
                shutil.copy(source_folder / f, destinationfile) #1522688014970187_1.jpg'

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = x['category_id'] - 1  # class
            line = cls, *(box)  # cls, box
            #with open((fn / f).with_suffix('.txt'), 'a') as file:
            labelfilename=imageidstr+'.txt'
            with open((fn / labelfilename), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

def convert_coco_json(source_dir, json_file='../coco/annotations/', save_dir='./cocoyolo', deleteold=True, copyimage=True, use_segments=False, cls91to80=False):
    save_dir = make_dirs(save_dir, deleteold)  # create output directory with labels and images folder
    coco80 = coco91_to_coco80_class() #len=91

    source_folder=Path(source_dir) #source images folder

    # Import json file
    fn = Path(save_dir) / 'labels' #/ json_file.stem.replace('instances_', '')  # folder name
    #fn.mkdir()
    with open(json_file) as f:
        data = json.load(f)
    
    # Create image dict
    #images = {'%g' % x['id']: x for x in data['images']}
    
    images = {'%g' % int(x['id']): x for x in data['images']}#our converted COCO 'id' field is str not int

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['iscrowd']:
            continue

        #img = images['%g' % x['image_id']]
        imageidstr=x['image_id']
        img = images['%g' % int(imageidstr)]
        h, w, f = img['height'], img['width'], img['file_name']

        #New: copy image files
        if copyimage == True:
            newimagefilename=f #imageidstr+'.jpg'
            shutil.copy(source_folder / f, save_dir / 'images' / newimagefilename)

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Segments
        if use_segments:
            segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
            s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
            line = cls, *(s if use_segments else box)  # cls, box or segments
            with open((fn / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

def image_folder2file(basefolder, folder='images/'):  # from utils import *; image_folder2file()
    # write a txt file listing all imaged in folder
    s = glob.glob(folder + '*.*')
    with open(folder[:-1] + '.txt', 'w') as file: #'/DATA5T2/Datasets/waymotrain200cocoyolo/images.txt'
        for l in s:
            #change absolute path to relative path
            relative_path = os.path.relpath(l, basefolder)
            file.write('./'+relative_path + '\n')  # write image list

def split_rows_simple(file='../data/sm4/out.txt'): 
    # splits one textfile into 3 smaller ones based upon train, test, val ratios
    with open(file) as f:
        lines = f.readlines()

    s = Path(file).suffix #.txt
    lines = sorted(list(filter(lambda x: len(x) > 0, lines)))
    i, j, k = split_indices(lines, train=0.8, test=0.1, validate=0.1)
    for k, v in {'train': i, 'test': j, 'val': k}.items():  # key, value pairs
        if v.any():
            new_file = file.replace(s, '_' + k + s) #'/DATA5T2/Datasets/waymotrain200cocoyolo/images_train.txt'
            with open(new_file, 'w') as f:
                f.writelines([lines[i] for i in v])

def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

if __name__ == '__main__':
    source = 'WAYMOCOCO'

    cocojsonfile=''
    valjsonfile='/DATA5T/Dataset/WaymoCOCO/annotations_val5filteredbig.json'
    trainjsonfile='/DATA5T/Dataset/WaymoCOCO/annotations_train20filteredbig.json' #annotations_train200new.json, annotations_trainallnew.json

    source_dir = '/DATA5T/Dataset/WaymoCOCO'
    waymojsonfile = '/DATA5T/Dataset/WaymoCOCO/annotations_train200new.json'
    save_dir='/DATA5T2/Datasets/waymotrain200cocoyolo'
    if source == 'WAYMOCOCO':
        #convert_waymococo_json(source_dir, waymojsonfile, save_dir, deleteold=True, copyimage=True)

        # Write *.names file
        #WAYMO_CLASSES =['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
        names = ['vehicle', 'pedestrian', 'sign', 'cyclist']  # preserves sort order
        with open(save_dir + '/data.names', 'w') as f:
            [f.write('%s\n' % a) for a in names]
        
        image_folder2file(save_dir, save_dir+'/images/')

        split_rows_simple(save_dir+'/images.txt')