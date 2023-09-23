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

   (mycondapy310) [010796032@cs004 detection]$ python mytrain.py --data-path="/data/cmpe249-fa23/COCOoriginal/" --dataset="coco" --model="fasterrcnn_resnet50_fpn_v2" --resume="" --test-only=True   

   IoU metric: bbox
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.301
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.489
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.350
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.187
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.480
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.377
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.438
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.186
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733

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
   Epoch0:
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