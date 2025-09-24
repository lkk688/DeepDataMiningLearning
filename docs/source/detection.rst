===============================
Object Detection Tutorial
===============================

.. _detection:

**Deep Learning for Object Detection: A Comprehensive Guide**

This document provides a comprehensive tutorial on object detection using deep learning models, covering various datasets, architectures, and training methodologies. The tutorial includes practical examples with COCO, KITTI, and Waymo datasets, as well as detailed implementations of YOLO models.

.. important::
   This tutorial is designed for researchers and practitioners working on computer vision and object detection tasks. It assumes basic knowledge of deep learning and PyTorch.

.. contents:: Table of Contents
   :local:
   :depth: 2

===============================
Author Information
===============================

:Author: **Kaikai Liu**, Associate Professor
:Institution: San José State University (SJSU)
:Department: Computer Engineering
:Email: kaikai.liu@sjsu.edu
:Website: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php

===============================
Overview
===============================

Object detection is a fundamental computer vision task that involves identifying and localizing objects within images. This tutorial covers:

.. grid:: 2

   .. grid-item-card:: Dataset Preparation
      :text-align: center

      Working with COCO, KITTI, and Waymo datasets

   .. grid-item-card:: Model Architectures
      :text-align: center

      Faster R-CNN, Custom R-CNN, and YOLO variants

   .. grid-item-card:: Training Strategies
      :text-align: center

      Single-GPU and multi-GPU training approaches

   .. grid-item-card:: Performance Evaluation
      :text-align: center

      Metrics interpretation and model comparison

.. note::
   The tutorial progresses from basic object detection concepts to advanced implementations, providing both theoretical background and practical code examples.

===============================
COCO Object Detection
===============================

The COCO (Common Objects in Context) dataset is one of the most widely used benchmarks for object detection. This section demonstrates training and evaluation procedures using Faster R-CNN and Custom R-CNN models.

Pre-trained Model Evaluation
-----------------------------

First, let's evaluate a pre-trained Faster R-CNN model on the COCO dataset to establish baseline performance:

.. code-block:: bash
   :caption: COCO Pre-trained Model Evaluation
   :linenos:

   (mycondapy310) [010796032@cs004 detection]$ python mytrain.py \
       --data-path="/data/cmpe249-fa23/COCOoriginal/" \
       --dataset="coco" \
       --model="fasterrcnn_resnet50_fpn_v2" \
       --resume="" \
       --test-only   

**Performance Results:**

.. code-block:: text
   :caption: Pre-trained Model Performance Metrics

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

.. tip::
   The pre-trained model achieves an mAP of **32.1%** on the COCO validation set, which serves as our baseline for comparison.

Custom R-CNN Training
----------------------

Training a custom R-CNN model from scratch on the COCO dataset:

**Training Command:**

.. code-block:: bash
   :caption: Custom R-CNN Training Configuration
   :linenos:

   python mytrain.py \
       --data-path "/data/cmpe249-fa23/COCOoriginal/" \
       --dataset "coco" \
       --model "customrcnn_resnet50" \
       --device "cuda:3" \
       --epochs 20 \
       --expname "0315coco" \
       --output-dir "/data/rnd-liu/output" \
       --annotationfile "" \
       --resume "/data/rnd-liu/output/coco/0315coco/model_12.pth"

**Results after 20 epochs:**

.. code-block:: text
   :caption: Custom R-CNN Performance (20 Epochs)

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

**Extended Training (40 epochs):**

.. code-block:: console

   python mytrain.py \
       --data-path "/data/cmpe249-fa23/COCOoriginal/" \
       --dataset "coco" \
       --model "customrcnn_resnet50" \
       --device "cuda:3" \
       --epochs 40 \
       --expname "0315coco" \
       --output-dir "/data/rnd-liu/output" \
       --annotationfile "" \
       --resume "/data/rnd-liu/output/coco/0315coco/model_20.pth"

**Results after 40 epochs:**

.. code-block:: text

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

.. tip::
   **Training Insights:**
   
   * The custom R-CNN model achieves 26.0% mAP after 40 epochs
   * Performance improvement from 20 to 40 epochs is modest (25.2% → 26.0%)
   * The model shows consistent performance across different object sizes
   * Consider implementing learning rate scheduling for better convergence


KITTI Object Detection
----------------------

The KITTI dataset is a popular benchmark for autonomous driving applications, containing real-world driving scenarios with various object classes including cars, pedestrians, and cyclists. This section demonstrates training procedures using transfer learning approaches.

Dataset Overview
~~~~~~~~~~~~~~~~

The KITTI dataset provides:

* **Real-world driving scenarios** from urban, residential, and highway environments
* **Multiple object classes**: Car, Van, Truck, Pedestrian, Person (sitting), Cyclist, Tram, Misc
* **3D annotations** with precise bounding boxes and occlusion levels
* **Challenging conditions**: Varying lighting, weather, and traffic density

Training Strategy: Transfer Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training follows a two-phase approach:

1. **Phase 1 (Epochs 1-36)**: Freeze backbone, train only detection head
2. **Phase 2 (Epochs 37-60)**: Unfreeze entire network for fine-tuning

**Training Command:**

.. code-block:: console

   (mycondapy310) [010796032@cs004 detection]$ python mytrain.py \
       --data-path="/data/cmpe249-fa23/torchvisiondata/Kitti/" \
       --dataset="kitti" \
       --model="fasterrcnn_resnet50_fpn_v2" \
       --resume="/data/cmpe249-fa23/trainoutput/kitti/model_36.pth" \
       --output-dir="/data/cmpe249-fa23/trainoutput"

Phase 1 Results (Epochs 1-36, Frozen Backbone)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

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

Phase 2 Results (Epochs 37-60, Unfrozen Network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

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

**Training Time:** 1:29:30

.. important::
   **Dramatic Performance Improvement:**
   
   * Phase 1 (frozen): 18.6% mAP
   * Phase 2 (unfrozen): 66.2% mAP
   * **Performance gain**: +47.6% mAP by unfreezing the backbone
   * This demonstrates the importance of fine-tuning pre-trained models

Final Evaluation
~~~~~~~~~~~~~~~~

**Evaluation Command:**

.. code-block:: console

   $ python mytrain.py \
       --data-path="/data/cmpe249-fa23/torchvisiondata/Kitti/" \
       --dataset="kitti" \
       --model="fasterrcnn_resnet50_fpn_v2" \
       --resume="/data/cmpe249-fa23/trainoutput/kitti/model_60.pth" \
       --output-dir="/data/cmpe249-fa23/trainoutput" \
       --test-only=True

**Final Test Results:**

.. code-block:: text

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

.. success::
   **Excellent Results:**
   
   * **Final mAP**: 75.8% (COCO metrics)
   * **AP@0.5**: 94.7% (very high precision at IoU=0.5)
   * **Strong performance** across all object sizes
   * The model generalizes well to the KITTI test set

Waymo-COCO Training Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training on a subset of Waymo dataset converted to COCO format:

**Training Command:**

.. code-block:: console

   $ python mytrain.py \
       --data-path="/data/cmpe249-fa23/WaymoCOCO/" \
       --dataset="waymococo"

**Progressive Training Results:**

*Epoch 8 (Frozen Backbone):*

.. code-block:: text

   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.218
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.319
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.247

*Epoch 32 (Unfrozen Network):*

.. code-block:: text

   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.274
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.406
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304

.. note::
   The Waymo-COCO experiments show consistent improvement with the two-phase training strategy, achieving 27.4% mAP after unfreezing the network.

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

YOLO Dataset Format and Structure
==================================

The YOLO (You Only Look Once) dataset format is a widely-used annotation format for object detection tasks. This section covers the dataset structure, annotation format, and conversion utilities.

Dataset Directory Structure
---------------------------

The YOLO dataset follows a specific directory structure that separates images and their corresponding label files:

.. code-block:: text

    datasets/
    ├── images/
    │   ├── train/              # Training images
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/                # Validation images
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── test/               # Test images (optional)
    │       ├── image1.jpg
    │       └── ...
    └── labels/
        ├── train/              # Training labels
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        ├── val/                # Validation labels
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        └── test/               # Test labels (optional)
            ├── image1.txt
            └── ...

**Key Points:**

- Each image has a corresponding text file with the same name
- Images and labels are stored in separate directories
- The directory structure supports train/validation/test splits

Downloading COCO Dataset in YOLO Format
---------------------------------------

Download COCO dataset and YOLO formatted labels via [getcoco.sh](/data/scripts/get_coco.sh). After downloading, you will see the following structure:

.. code-block:: console

   Dataset/coco$ ls
   annotations  labels   README.txt        train2017.cache  val2017.cache
   images       LICENSE  test-dev2017.txt  train2017.txt    val2017.txt
   
   Dataset/coco$ ls images/
   test2017  train2017  val2017
   
   Dataset/coco$ cat labels/train2017/000000436300.txt
   5 0.527578 0.541663 0.680750 0.889628
   0 0.982781 0.696762 0.033719 0.174020
   0 0.169938 0.659901 0.020406 0.080844
   0 0.093156 0.685223 0.015031 0.081315

Label Format Specification
--------------------------

YOLO uses a simple text-based annotation format where each line represents one object:

.. code-block:: text

    class_id center_x center_y width height

**Format Details:**

- **class_id**: Integer representing the object class (0-indexed)
- **center_x**: Normalized x-coordinate of bounding box center (0.0 - 1.0)
- **center_y**: Normalized y-coordinate of bounding box center (0.0 - 1.0)
- **width**: Normalized width of bounding box (0.0 - 1.0)
- **height**: Normalized height of bounding box (0.0 - 1.0)

.. important::
   **Format Comparison:**
   
   - **YOLO format**: (class x_center y_center width height) - normalized values (0-1)
   - **COCO format**: [top_left_x, top_left_y, width, height] - pixel coordinates

Dataset Conversion Tools
------------------------

**COCO to YOLO Conversion:**

If you have the original COCO annotation file (.json), use the conversion script: [cocojsontoyolo.py](/data/cocojsontoyolo.py). Add the COCO .json file path in the main function of this file.

**Conversion Options:**

1. **Direct conversion**: Convert other dataset formats directly to YOLO format
2. **Two-step conversion**: Convert to standard COCO JSON format first, then use [cocojsontoyolo.py](/data/cocojsontoyolo.py)
3. **Waymo dataset**: For Waymo dataset conversion, check the [WaymoObjectDetection](https://github.com/lkk688/WaymoObjectDetection) repository

.. tip::
   **Best Practices:**
   
   - Ensure image and label filenames match exactly (except extensions)
   - Verify all coordinate values are normalized (0.0 - 1.0)
   - Check class IDs are 0-indexed and consecutive
   - Validate bounding box coordinates don't exceed image boundaries

YOLOv8 Custom Training and Implementation
=========================================

YOLOv8 represents the latest evolution in the YOLO (You Only Look Once) family of object detection models, offering state-of-the-art performance with improved accuracy and efficiency. This section demonstrates custom training implementation and performance evaluation.

Model Overview
--------------

YOLOv8 introduces several key improvements over previous versions:

- **Anchor-free detection**: Eliminates the need for predefined anchor boxes
- **New backbone architecture**: Enhanced feature extraction with C2f modules
- **Improved loss functions**: Task-aligned learning for better optimization
- **Multiple model sizes**: From nano (YOLOv8n) to extra-large (YOLOv8x) variants

Custom Training Implementation
------------------------------

**Multi-GPU Training Command:**

.. code-block:: bash

    # YOLOv8x training on WaymoCOCO dataset with multi-GPU setup
    torchrun --nproc_per_node=2 DeepDataMiningLearning/detection/mytrain_yolo.py \
             --data-path='/data/cmpe249-fa23/waymotrain200cocoyolo/' \
             --dataset='yolo' --model='yolov8' --scale='x' \
             --ckpt='/data/cmpe249-fa23/modelzoo/yolov8x_statedicts.pt' \
             --batch-size=8 --trainable=0 --multigpu=True

**Training Configuration:**

- **Model**: YOLOv8x (extra-large variant)
- **Dataset**: WaymoCOCO in YOLO format
- **Multi-GPU**: 2 GPUs with distributed training
- **Batch Size**: 8 samples per GPU
- **Backbone**: Frozen (trainable=0) for transfer learning

Training Performance Results
----------------------------

The following results demonstrate YOLOv8x performance across different training epochs on the WaymoCOCO dataset:

**Epoch 4 Results:**

.. list-table:: Performance Metrics - Epoch 4
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - All Objects
     - Small Objects
     - Medium Objects
   * - AP@0.5:0.95
     - 0.313
     - 0.199
     - 0.646
   * - AP@0.5
     - 0.465
     - --
     - --
   * - AP@0.75
     - 0.336
     - --
     - --
   * - AR@0.5:0.95 (maxDets=100)
     - 0.346
     - 0.227
     - 0.692

**Epoch 22 Results:**

.. list-table:: Performance Metrics - Epoch 22
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - All Objects
     - Small Objects
     - Medium Objects
   * - AP@0.5:0.95
     - 0.319
     - 0.204
     - 0.664
   * - AP@0.5
     - 0.470
     - --
     - --
   * - AP@0.75
     - 0.345
     - --
     - --
   * - AR@0.5:0.95 (maxDets=100)
     - 0.350
     - 0.228
     - 0.708

**Epoch 26 Results:**

.. list-table:: Performance Metrics - Epoch 26
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - All Objects
     - Small Objects
     - Medium Objects
   * - AP@0.5:0.95
     - 0.356
     - 0.242
     - 0.691
   * - AP@0.5
     - 0.533
     - --
     - --
   * - AP@0.75
     - 0.381
     - --
     - --
   * - AR@0.5:0.5 (maxDets=100)
     - 0.387
     - 0.274
     - 0.732

**Final Results - Epoch 60:**

.. list-table:: Performance Metrics - Epoch 60 (Final)
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - All Objects
     - Small Objects
     - Medium Objects
   * - AP@0.5:0.95
     - **0.381**
     - 0.262
     - 0.727
   * - AP@0.5
     - **0.573**
     - --
     - --
   * - AP@0.75
     - **0.405**
     - --
     - --
   * - AR@0.5:0.95 (maxDets=100)
     - **0.415**
     - 0.298
     - 0.766

**Performance Analysis:**

- **Overall mAP improvement**: From 0.313 to 0.381 (21.7% increase)
- **AP@0.5 improvement**: From 0.465 to 0.573 (23.2% increase)
- **Strong performance on medium/large objects**: Consistent 0.7+ AP scores
- **Steady convergence**: Gradual improvement across all metrics

Ultralytics Integration
-----------------------

**Quick Start with Ultralytics:**

For rapid prototyping and experimentation, use the official Ultralytics implementation:

**Installation:**

.. code-block:: bash

    # Clone and install custom YOLOv8 implementation
    git clone https://github.com/lkk688/myyolov8.git
    cd myyolov8
    pip install -e .

**Model Loading and Usage:**

.. code-block:: python

    from ultralytics import YOLO
    
    # Load model from YAML configuration
    model = YOLO('yolov8n.yaml')
    
    # Load pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(data='coco128.yaml', epochs=3)

**Reference:** `Ultralytics YOLOv8 Documentation <https://docs.ultralytics.com/quickstart/>`_

Model Architecture and Implementation Details
--------------------------------------------

**Model Loading Process:**

.. code-block:: python

    # Model initialization flow
    class Model from ultralytics.engine.model.py:
        def _new(model):
            cfg_dict = yaml_model_load(cfg)  # from ultralytics.nn.tasks.py
            self.model = (model or self._smart_load('model'))
            
            class DetectionModel(BaseModel):  # from ultralytics.nn.tasks.py
                self.model, self.save = parse_model(deepcopy(self.yaml))
                self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # dict 0~79
        
        def _load():  # load weights when calling model = YOLO('yolov8n.pt')
            def attempt_load_one_weight():
                def torch_safe_load(weight):
                    return torch.load(file, map_location='cpu')
                model = ckpt['model'].eval()

**Inference Pipeline:**

.. code-block:: python

    # Inference flow through the model
    def predict(source, stream, **kwargs):
        # Setup predictor
        self.predictor.setup_model(model=self.model, verbose=is_cli)
        self.model = AutoBackend(model)  # from ultralytics.nn.autobackend.py
        
        # Preprocessing
        im = self.preprocess(im0s)  # [1080, 810, 3] -> [1, 3, 640, 480]
        # LetterBox -> BHWC to BCHW -> normalize /255
        
        # Forward pass
        preds = self.inference(im, *args, **kwargs)
        # Model forward: [1, 3, 640, 480] -> [1, 84, 6300]
        # 84 = 4 bbox + 80 classes, 6300 = 80*60 + 40*30 + 20*15
        
        # Post-processing
        self.results = self.postprocess(preds, im, im0s)
        # Non-max suppression -> scale boxes -> final detections

**Forward Pass Architecture:**

.. code-block:: text

    Input: [1, 3, 640, 480]
    ├── Conv -> [1, 16, 320, 240]
    ├── Conv -> [1, 32, 160, 120]
    ├── C2f -> [1, 32, 160, 120]
    ├── Conv -> [1, 64, 80, 60]
    ├── C2f -> [1, 64, 80, 60]
    ├── Conv -> [1, 128, 40, 30]
    ├── C2f -> [1, 128, 40, 30]
    ├── Conv -> [1, 256, 20, 15]
    ├── C2f -> [1, 256, 20, 15]
    ├── SPPF -> [1, 256, 20, 15]
    ├── Upsample -> [1, 256, 40, 30]
    ├── Concat -> [1, 384, 40, 30]
    ├── C2f -> [1, 128, 40, 30]
    ├── Upsample -> [1, 128, 80, 60]
    ├── Concat -> [1, 192, 80, 60]
    ├── C2f -> [1, 64, 80, 60]
    ├── Conv -> [1, 64, 40, 30]
    ├── Concat -> [1, 192, 40, 30]
    ├── C2f -> [1, 128, 40, 30]
    ├── Conv -> [1, 128, 20, 15]
    ├── Concat -> [1, 384, 20, 15]
    ├── C2f -> [1, 256, 20, 15]
    └── Detect -> Output: [1, 84, 6300]

**Detection Head Output:**

.. code-block:: python

    # Detection head processes three feature maps
    inputs: [1, 64, 80, 60], [1, 128, 40, 30], [1, 256, 20, 15]
    
    # Each input processed through cv2 and cv3 branches
    outputs: [1, 144, 80, 60], [1, 144, 40, 30], [1, 144, 20, 15]
    
    # Concatenate and split
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # [1, 144, 6300]
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # box: [1, 64, 6300], cls: [1, 80, 6300]
    
    # Final output
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    y = torch.cat((dbox, cls.sigmoid()), 1)  # [1, 84, 6300]

Training Process Implementation
------------------------------

**Data Loading:**

.. code-block:: python

    # YOLODataset format
    def get_labels(self):
        """Custom format for YOLO training data
        Output format:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """

**Training Loop:**

.. code-block:: python

    # Training process in DetectionTrainer
    class DetectionTrainer(BaseTrainer):
        def preprocess_batch(self, batch):
            """Preprocesses a batch of images by scaling and converting to float."""
            batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
            if self.args.multi_scale:
                imgs = batch["img"]
                sz = (
                    random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                    // self.stride
                    * self.stride
                )  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                    ]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                batch["img"] = imgs
            return batch

        def get_model(self, cfg=None, weights=None, verbose=True):
            """Return a YOLO detection model."""
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
            if weights:
                model.load(weights)
            return model

        def get_validator(self):
            """Returns a DetectionValidator for YOLO model validation."""
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return yolo.detect.DetectionValidator(
                self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
            )

        def criterion(self, preds, batch):
            """Compute loss with v8DetectionLoss."""
            if not hasattr(self, "compute_loss"):
                self.compute_loss = v8DetectionLoss(self.model)  # init loss class
            return self.compute_loss(preds, batch)

===============================
References and Resources
===============================

Academic Papers
---------------

.. bibliography::

   - **YOLO Series:**
     - Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.
     - Redmon, J., & Farhadi, A. "YOLO9000: Better, Faster, Stronger." CVPR 2017.
     - Redmon, J., & Farhadi, A. "YOLOv3: An Incremental Improvement." arXiv 2018.
     - Bochkovskiy, A., et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection." CVPR 2020.
     - Jocher, G., et al. "YOLOv5: A State-of-the-Art Real-Time Object Detection System." 2020.

   - **R-CNN Series:**
     - Girshick, R., et al. "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation." CVPR 2014.
     - Girshick, R. "Fast R-CNN." ICCV 2015.
     - Ren, S., et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NIPS 2015.

   - **Datasets:**
     - Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
     - Geiger, A., et al. "Vision meets Robotics: The KITTI Dataset." IJRR 2013.
     - Sun, P., et al. "Scalability in Perception for Autonomous Driving: Waymo Open Dataset." CVPR 2020.

Official Documentation
----------------------

.. list-table:: Framework and Library Resources
   :widths: 30 70
   :header-rows: 1

   * - Resource
     - Link
   * - Ultralytics YOLOv8
     - https://docs.ultralytics.com/
   * - PyTorch Vision
     - https://pytorch.org/vision/stable/index.html
   * - COCO Dataset
     - https://cocodataset.org/
   * - KITTI Dataset
     - http://www.cvlibs.net/datasets/kitti/
   * - Waymo Open Dataset
     - https://waymo.com/open/

Code Repositories
-----------------

.. code-block:: text

   # Local Implementation Files
   ├── detection/
   │   ├── tal.py                    # Task Aligned Learning implementation
   │   ├── lossv8.py                # YOLOv8 loss functions
   │   ├── complete_loss_test.py    # Loss function testing
   │   └── mytrain.py               # Training script
   
   # External Repositories
   ├── Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
   ├── PyTorch Vision: https://github.com/pytorch/vision
   └── COCO API: https://github.com/cocodataset/cocoapi

.. note::
   This tutorial provides comprehensive coverage of object detection methodologies, from traditional R-CNN approaches to modern YOLO implementations. For the latest updates and additional resources, please refer to the official documentation and repositories listed above.

.. tip::
   **Getting Started Quickly:**
   
   1. Install required dependencies: ``pip install ultralytics torch torchvision``
   2. Download datasets using the provided scripts
   3. Start with pre-trained models for evaluation
   4. Gradually move to custom training configurations
   5. Experiment with different architectures and hyperparameters

===============================

*Last updated: 2024 | Author: Kaikai Liu, SJSU*
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        
        def _do_train(self):
            batch = self.preprocess_batch(batch)
            self.loss, self.loss_items = self.model(batch)
            # loss_items = [box_loss, cls_loss, dfl_loss]
            self.scaler.scale(self.loss).backward()

**Batch Collation:**

.. code-block:: python

    # Collate function for batch processing
    def collate_fn(batch):
        # Input: 16 image files, each dict with:
        # 'im_file', 'ori_shape', 'resized_shape', 'img' [3, 640, 640], 
        # 'cls' [27,1], 'bboxes' [27,4], 'batch_idx' [27]
        
        # Output: Combined batch
        # 'img' [16, 3, 640, 640], 'cls' [391, 1], 'bboxes' [391, 4], 
        # 'batch_idx' [391] (0~16 identify objects for each image)

**Loss Computation:**

.. code-block:: python

    # v8DetectionLoss implementation
    class v8DetectionLoss:
        def __call__(self, preds, batch):
            # Compute box loss, classification loss, and DFL loss
            # Returns total loss and individual loss components