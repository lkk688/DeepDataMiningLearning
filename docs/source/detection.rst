detection
==========

.. _detection:

Author:
   * *Kaikai Liu*, Associate Professor, SJSU
   * **Email**: kaikai.liu@sjsu.edu
   * **Web**: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php


Kitti Object Detection
----------------------

Perform Kitti object detection training, model saved in "/data/cmpe249-fa23/trainoutput/kitti"

.. code-block:: console

   (mycondapy310) [010796032@cs004 detection]$ python mytrain.py --dataset='kitti', --model='fasterrcnn_resnet50_fpn_v2', --data_path="/data/cmpe249-fa23/torchvisiondata/Kitti/", --output-dir="/data/cmpe249-fa23/trainoutput"
   Epoch1-36 (freeze the FasterRCNN except the header)
   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.186
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.277
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.210
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.193
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.239
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.141
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.206
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.206
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.212
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.264
   Epoch: 37-60 (unfreeze the FasterRCNN)
   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.662
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.860
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.760
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.705
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.680
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.482
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.715
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.718
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.746
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.722
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
   Training time 1:29:30