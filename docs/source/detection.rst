detection
==========

.. _detection:

Author:
   * *Kaikai Liu*, Associate Professor, SJSU
   * **Email**: kaikai.liu@sjsu.edu
   * **Web**: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php


COCO Object Detection
---------------------

.. code-block:: console

   (mycondapy310) [010796032@cs004 detection]$ python mytrain.py --data-path="/data/cmpe249-fa23/COCOoriginal/" --dataset="coco" --model="fasterrcnn_resnet50_fpn_v2" --resume="" --test-only   

   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.321
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.552
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.340
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.385
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.469
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.214
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767

Train "customrcnn_resnet50" based on COCO dataset:

.. code-block:: console

   python mytrain.py --data-path "/data/cmpe249-fa23/COCOoriginal/" --dataset "coco" --model "customrcnn_resnet50" --device "cuda:3" --epochs 20 --expname "0315coco"  --output-dir "/data/rnd-liu/output" --annotationfile "" --resume "/data/rnd-liu/output/coco/0315coco/model_12.pth"
   After 20 epoches
   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.460
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.245
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.157
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.295
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.296
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.409
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.277
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.480
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.520

   python mytrain.py --data-path "/data/cmpe249-fa23/COCOoriginal/" --dataset "coco" --model "customrcnn_resnet50" --device "cuda:3" --epochs 40 --expname "0315coco"  --output-dir "/data/rnd-liu/output" --annotationfile "" --resume "/data/rnd-liu/output/coco/0315coco/model_20.pth"
   After 40 epoches
   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.472
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.307
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.244
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.414
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.287
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.518


Kitti Object Detection
----------------------

Perform Kitti object detection training, model saved in "/data/cmpe249-fa23/trainoutput/kitti"

.. code-block:: console

   (mycondapy310) [010796032@cs004 detection]$ python mytrain.py --data-path="/data/cmpe249-fa23/torchvisiondata/Kitti/" --dataset="kitti" --model="fasterrcnn_resnet50_fpn_v2" --resume="/data/cmpe249-fa23/trainoutput/kitti/model_36.pth" --output-dir="/data/cmpe249-fa23/trainoutput"
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

Perform Kitti evaluation:

.. code-block:: console

   $ python mytrain.py --data-path="/data/cmpe249-fa23/torchvisiondata/Kitti/" --dataset="kitti" --model="fasterrcnn_resnet50_fpn_v2" --resume="/data/cmpe249-fa23/trainoutput/kitti/model_60.pth" --output-dir="/data/cmpe249-fa23/trainoutput" --test-only=True
   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.758
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.947
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.947
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.800
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.746
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.556
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.749
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.772
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.800
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.667
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767

Perform WaymoCOCO training:

.. code-block:: console

   $ python mytrain.py --data-path="/data/cmpe249-fa23/WaymoCOCO/" --dataset="waymococo"
   Epoch8: freeze=True
   DONE (t=2.94s).
   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.218
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.319
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.247
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.249
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.103
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.227
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.234
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.025
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.275
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
   
   Epoch12: freeze=True
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.216
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.313
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.244
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.244
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.103
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.225
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.023
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.270
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.646

   Epoch32: freeze=False
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.274
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.406
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.324
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.121
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.276
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.293
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.714

CustomRCNN with Resnet152 backbone training with multi-GPU

.. code-block:: console

   (mycondapy310) [010796032@cs003 detection]$ torchrun --nproc_per_node=4 mytrain.py --batch-size=32
   Epoch0: trainable=0
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.162
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.355
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.124
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.401
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.081
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.210
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.248
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.311
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.518

   Epoch4:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.239
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.455
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.230
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.285
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.563
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.113
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.270
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.094
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.635

   Epoch8:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.250
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.465
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.242
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.298
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.116
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.280
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.321
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.097
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.657

   Epoch12:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.259
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.472
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.256
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.311
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.597
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.119
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.287
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.099
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.413
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668

   Epoch16:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.052
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.315
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.614
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.121
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.291
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.104
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675

   Epoch20 (stop) trainable=0
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.478
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.261
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.053
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.314
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.121
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.288
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.105
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.673

   (mycondapy310) [010796032@cs003 detection]$ torchrun --nproc_per_node=4 mytrain.py --batch-size=8 --trainable=2 --resume="/data/cmpe249-fa23/trainoutput/waymococo/0923/model_20.pth"
   Epoch24
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.291
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.291
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.340
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.128
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.308
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722

   Epoch32
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.505
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.289
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.054
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.339
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.129
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.304
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.344
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.102
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725

   torchrun --nproc_per_node=4 mytrain.py --batch-size=8 --trainable=4 --resume="/data/cmpe249-fa23/trainoutput/waymococo/0923/model_32.pth"
   
   Epoch36
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.305
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.310
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.132
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.318
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.112
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724

   Epoch40
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.293
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.299
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.351
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.128
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.307
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.431
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731

GPU Utilization:

.. code-block:: console

   Sat Sep 23 09:45:39 2023       
   +---------------------------------------------------------------------------------------+
   | NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
   |-----------------------------------------+----------------------+----------------------+
   | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                                         |                      |               MIG M. |
   |=========================================+======================+======================|
   |   0  NVIDIA A100-PCIE-40GB           On | 00000000:17:00.0 Off |                    0 |
   | N/A   62C    P0               76W / 250W|  33053MiB / 40960MiB |     65%      Default |
   |                                         |                      |             Disabled |
   +-----------------------------------------+----------------------+----------------------+
   |   1  NVIDIA A100-PCIE-40GB           On | 00000000:65:00.0 Off |                    0 |
   | N/A   62C    P0               70W / 250W|  37191MiB / 40960MiB |      5%      Default |
   |                                         |                      |             Disabled |
   +-----------------------------------------+----------------------+----------------------+
   |   2  NVIDIA A100-PCIE-40GB           On | 00000000:CA:00.0 Off |                    0 |
   | N/A   59C    P0               78W / 250W|  37151MiB / 40960MiB |      4%      Default |
   |                                         |                      |             Disabled |
   +-----------------------------------------+----------------------+----------------------+
   |   3  NVIDIA A100-PCIE-40GB           On | 00000000:E3:00.0 Off |                    0 |
   | N/A   61C    P0               81W / 250W|  37131MiB / 40960MiB |      4%      Default |
   |                                         |                      |             Disabled |
   +-----------------------------------------+----------------------+----------------------+
                                                                                          
   +---------------------------------------------------------------------------------------+
   | Processes:                                                                            |
   |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
   |        ID   ID                                                             Usage      |
   |=======================================================================================|
   |    0   N/A  N/A     18464      C   ...conda3/envs/mycondapy310/bin/python    33050MiB |
   |    1   N/A  N/A     18465      C   ...conda3/envs/mycondapy310/bin/python    37188MiB |
   |    2   N/A  N/A     18466      C   ...conda3/envs/mycondapy310/bin/python    37148MiB |
   |    3   N/A  N/A     18467      C   ...conda3/envs/mycondapy310/bin/python    37128MiB |
   +---------------------------------------------------------------------------------------+

YOLO dataset
-------------
Download COCO dataset and YOLO formated label via [getcoco.sh](/data/scripts/get_coco.sh), after downloading you will see the following folders

.. code-block:: console

   Dataset/coco$ ls
   annotations  labels   README.txt        train2017.cache  val2017.cache
   images       LICENSE  test-dev2017.txt  train2017.txt    val2017.txt
   /Dataset/coco$ ls images/
   test2017  train2017  val2017
   /Dataset/coco$ cat labels/train2017/000000436300.txt
   5 0.527578 0.541663 0.680750 0.889628
   0 0.982781 0.696762 0.033719 0.174020
   0 0.169938 0.659901 0.020406 0.080844
   0 0.093156 0.685223 0.015031 0.081315

   
The label format is YOLO format, not the original COCO annotation format. Each row in the .txt label file is (class x_center y_center width height) format, all these are in normalized xywh format (from 0 - 1). The COCO box format is [top left x, top left y, width, height] in pixels.

If you have the original COCO annotation file (.json), I added this python code to do the conversion: [cocojsontoyolo.py](/data/cocojsontoyolo.py). You can add the COCO .json file path in the main function of this file. If you have other dataset format, you can direct convert them to YOLO format, you can also convert them to standard COCO json format, then use [cocojsontoyolo.py](/data/cocojsontoyolo.py) to convert to YOLO format. If you want to use the Waymo dataset, you can check my [WaymoObjectDetection](https://github.com/lkk688/WaymoObjectDetection) repository.

YOLOv8
-------
Our custom YOLOv8 training

.. code-block:: console

   $ torchrun --nproc_per_node=2 DeepDataMiningLearning/detection/mytrain_yolo.py --data-path='/data/cmpe249-fa23/waymotrain200cocoyolo/' --dataset='yolo' --model='yolov8' --scale='x' --ckpt='/data/cmpe249-fa23/modelzoo/yolov8x_statedicts.pt' --batch-size=8 --trainable=0 --multigpu=True
   Epoch4:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.313
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.465
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.336
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.199
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.124
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.346
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.692
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.785

   $ torchrun --nproc_per_node=2 --master_port=25641 DeepDataMiningLearning/detection/mytrain_yolo.py --data-path='/data/cmpe249-fa23/waymotrain200cocoyolo/' --dataset='yolo' --model='yolov8' --scale='x' --ckpt='/data/cmpe249-fa23/modelzoo/yolov8x_statedicts.pt' --batch-size=8 --trainable=0 --multigpu=True --resume='/data/cmpe249-fa23/trainoutput/yolo/1004/model_21.pth' --expname='yolov8x1005'
   Epoch22:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.319
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.345
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.204
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.664
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.726
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.121
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.305
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.228
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.708
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.796

   Epoch26:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.356
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.381
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.691
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.694
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.137
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.336
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.732
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755

   Epoch36:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.545
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.708
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.761
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.143
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.348
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.287
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.745
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.806

   Epoch60:
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.573
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.405
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.262
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.727
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.151
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.360
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.298
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.766
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.816

https://docs.ultralytics.com/quickstart/

.. code-block:: console

   ~/Developer$ git clone https://github.com/lkk688/myyolov8.git
   :~/Developer/myyolov8$ pip install -e .

.. code-block:: console

   model = YOLO('yolov8n.yaml')
   model = YOLO('yolov8n.pt')
      class Model from ultralytics\engine\model.py
      model='yolov8n.yaml'
      def _new(model)
         cfg_dict = yaml_model_load(cfg) #from ultralytics\nn\tasks.py
         self.model = (model or self._smart_load('model'))
            self.task_map[self.task][key] in self.task_map[self.task][key]

            class DetectionModel(BaseModel) #from ultralytics\nn\tasks.py
               self.model, self.save = parse_model(deepcopy(self.yaml) #yaml is dict
               self.names = {i: f'{i}' for i in range(self.yaml['nc'])} #dict 0~79
         
         self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
      def _load #load weights when call model = YOLO('yolov8n.pt')
         def attempt_load_one_weight
            def torch_safe_load(weight)
               torch.load(file, map_location='cpu')
            model = ckpt['model']
            model = model.eval()
      .model = DetectionModel

   from ultralytics.utils import DEFAULT_CFG

   Inference
      ultralytics\engine\model.py
      self.predict(source, stream, **kwargs)
         self.task_map[self.task][key] in self._smart_load('predictor')
            class BasePredictor from ultralytics\engine\predictor.py
         self.predictor.setup_model(model=self.model, verbose=is_cli)
            self.model = AutoBackend(model in ultralytics\engine\predictor.py
               ultralytics\nn\autobackend.py
               nn_module=True
               model = model.fuse(verbose=verbose) #fuse=True
         def stream_inference
            im = self.preprocess(im0s) #im0s list of array(1080, 810, 3)->[1, 3, 640, 480]
               def pre_transform
                  LetterBox
               (1, 640, 480, 3)->(1, 3, 640, 480) BHWC to BCHW
               totensor torch.Size([1, 3, 640, 480])
               im /= 255
            preds = self.inference(im, *args, **kwargs)
               self.model(im -> def forward in ultralytics\nn\autobackend.py
                  y = self.model(im tensor[1, 3, 640, 480]
                     self.predict->_predict_once in class BaseModel(nn.Module) (ultralytics\nn\tasks.py)
                        Conv(x)->[1, 16, 320, 240]
                        Conv(x)->[1, 32, 160, 120]
                        C2f->[1, 32, 160, 120]
                        Conv(x)->[1, 64, 80, 60]
                        C2f->[1, 64, 80, 60]
                        Conv(x)->[1, 128, 40, 30]
                        C2f->[1, 128, 40, 30]
                        Conv(x)->[1, 256, 20, 15]
                        C2f->[1, 256, 20, 15]
                        SPPF->[1, 256, 20, 15]
                        Upsample->[1, 256, 40, 30]
                        Concat->[1, 384, 40, 30]
                        C2f->[1, 128, 40, 30]
                        Upsample->[1, 128, 80, 60]
                        Concat(two tensors in [1, 128, 80, 60] [1, 64, 80, 60])->[1, 192, 80, 60]
                        C2f->[1, 64, 80, 60]
                        Conv->[1, 64, 40, 30]
                        Concat->[1, 192, 40, 30]
                        C2f->[1, 128, 40, 30]
                        Conv->[1, 128, 20, 15]
                        Concat(-1,9)->[1, 384, 20, 15]
                        C2f->[1, 256, 20, 15]
                        Detect[15, 18, 21] three inputs: [1, 64, 80, 60], [1, 128, 40, 30], [1, 256, 20, 15]
                           def forward in ultralytics\nn\modules\head.py
                           nl=3
                           x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
                           [1, 144, 80, 60], [1, 144, 40, 30], [1, 144, 20, 15]
                           6300=80*60+40*30+20*15
                           anchors [2, 6300], strides [1, 6300], shape: [1, 64, 80, 60]
                           x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) ->[1, 144, 6300]
                           box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
                           box: [1, 64, 6300], cls: [1, 80, 6300]
                           dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
                           dbox: [1, 4, 6300]
                           y = torch.cat((dbox, cls.sigmoid()), 1)
                           output: y: [1, 84, 6300] and x ([1, 144, 80, 60], [1, 144, 40, 30], [1, 144, 20, 15])
                  if isinstance(y, (list, tuple)):
                     return [self.from_numpy(x) for x in y] in autobackend
            preds = self.inference(im, *args, **kwargs) return (inference_out, loss_out)
            preds tensor [1, 84, 6300] and x ([1, 144, 80, 60], [1, 144, 40, 30], [1, 144, 20, 15])
            84=4 bbox + 80 classes
            self.results = self.postprocess(preds, im, im0s)
               preds = ops.non_max_suppression(preds #classes=None
                  prediction = prediction[0]  # select only inference output [1, 84, 6300]
               preds: list of one tensor[6, 6]
               pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

Load images

.. code-block:: console

   class LoadImages: in ultralytics\data\loaders.py
      def __next__(self):
         im0 = cv2.imread(path) #(1080, 810, 3)
         batch = return [path], [im0]

   ultralytics\engine\predictor.py

.. code-block:: console

   def collate_fn(batch):  in ultralytics\data\dataset.py
   input batch 16 imagefile list, each item is a dict with:
    'im_file', 'ori_shape', 'resized_shape', 'img' [3, 640, 640], 'cls' [27,1], 'bboxes' [27,4], 'batch_idx' [27]
   new_batch['batch_idx'][i] += i
   output new_batch: 'img' [16, 3, 640, 640], 'cls' [391, 1], 'bboxes' [391, 4], 'batch_idx' [391] (0~16 identify objects for each image)

   class v8DetectionLoss: in ultralytics\utils\loss.py
      def __call__(self, preds, batch):
   
   self.loss, self.loss_items = self.model(batch) in ultralytics\engine\trainer.py
   self.tloss = loss_items=[0.8499, 1.8445, 1.1006]
   self.scaler.scale(self.loss).backward() 

   def get_labels(self): in ultralytics\data\base.py
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """

Training Process

.. code-block:: console

   results = model.train(data='coco128.yaml', epochs=3) call def train in ultralytics\engine\model.py
   ultralytics\models\yolo\detect\train.py

   class DetectionTrainer(BaseTrainer) in ultralytics\models\yolo\detect\train.py
      preprocess_batch(self, batch):
         batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
      build_yolo_dataset return YOLODataset in ultralytics\data\build.py
      build_dataloader(dataset, batch_size, workers, shuffle, rank) return InfiniteDataLoader in ultralytics\data\build.py

   class BaseTrainer in ultralytics\engine\trainer.py
      def _setup_train
      def _do_train
         batch = self.preprocess_batch(batch)
         self.loss, self.loss_items = self.model(batch)

v8_transforms

build_transforms in YOLODataset