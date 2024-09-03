Welcome to DeepDataMiningLearning documentation!
=================================================

Author:
   * *Kaikai Liu*, Associate Professor, SJSU
   * **Email**: kaikai.liu@sjsu.edu
   * **Web**: http://www.sjsu.edu/cmpe/faculty/tenure-line/kaikai-liu.php

Check out the :ref:`CondaEnv` section for Conda environment setup; Check out the :ref:`hpc` section for SJSU CoE HPC environment setup

.. note::

   This project is under active development.

If you find the tutorials helpful and would like to cite them, you can use the following bibtex::

   @misc{kliu2024ddml,
      title        = {{DeepDataMiningLearning Tutorials}},
      author       = {Kaikai Liu},
      year         = 2024,
      howpublished = {\url{https://deepdatamininglearning.readthedocs.io/}}
   }

Contents
--------
There are three main ways of running the notebooks we recommend:

- **Local Machine**: You can download our sample code from `Github <https://github.com/lkk688/DeepDataMiningLearning/tree/main>`_. Following the following sections to setup your local AI system (choose the Machine type, CPU or GPU version depending on your system).

- **Google Colab**: If you prefer to run the code on a different platform than your own computer, or want to experiment with GPU support, we recommend using `Google Colab <https://colab.research.google.com/notebooks/intro.ipynb#recent=true>`_. Each notebook on this documentation website has a badge with a link to open it on Google Colab. Remember to enable GPU support before running the notebook (:code:`Runtime -> Change runtime type`). If using Colab, changes will  be lost after timeout or when closed your session. You need to manually save the data to your local computer or your Google Drive.

- **SJSU CoE HPC**: If you want to save your large dataset and train your own (larger) neural networks for a longer period (longer than Colab's timeout), you can make use of our SJSU CoE HPC cluster. The setup of your HPC workspace is documented beblow in the HPC section.

.. toctree::
   :caption: AI System Setups
   :maxdepth: 2

   source/MacDev
   source/MacML
   source/python
   source/linux
   source/CondaEnv
   source/HPC2
   source/Windows

.. toctree::
   :caption: Pytorch Tutorial
   :maxdepth: 2

   notebooks/CMPE_pytorch1_2024Fall
   notebooks/CMPE_pytorch2_2024Fall
   notebooks/CMPE_pytorch3_2024Fall
   notebooks/CMPE_pytorch4_2024Fall
   notebooks/CMPE_pytorch5_2024Fall_imageclassification
   notebooks/CMPE_pytorch6_2024Fall_mlp2resnet
   notebooks/CMPE_pytorch7_2024Fall_torchvision
   notebooks/CMPE_pytorch8_2024Fall_timm
   notebooks/CMPE_pytorch9_2024Fall_inferenceoptimization
   notebooks/CMPE_pytorch10_2024Fall_huggingfaceimage


.. toctree::

   source/detection
   source/container

