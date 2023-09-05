# DeepDataMiningLearning
Data mining, machine learning, and deep learning sample codes for SJSU CMPE255 Data Mining ([Fall2023 SJSU Official Syllabus](https://sjsu.campusconcourse.com/view_syllabus?course_id=22871&public_mode=1)) and CMPE258 Deep Learning ([Fall2023 SJSU Official Syllabus](https://sjsu.campusconcourse.com/view_syllabus?course_id=26399&public_mode=1)).
* Some google colab examples need SJSU google account to view)
* Large language Models (LLMs) part is newly added
* You can also view the documents in: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)

## Setups
Install this python package (optional) via

```bash
% python3 -m pip install flit
% flit install --symlink
```
ref "docs/python.rst" for detailed python package description

Open the Jupyter notebook in local machine:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

## Sphinx docs

Activate python virtual environment, you can use 'sphinx-build' command to build the document

```bash
   % pip install -r requirements.txt
   (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % sphinx-build docs ./docs/build
   #check the integrity of all internal and external links:
   (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % sphinx-build docs -W -b linkcheck -d docs/build/doctrees docs/build/html
```
The generated html files are in the folder of "build". You can also view the documents in: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)

## Python Data Analytics
Basic python tutorials, numpy, Pandas, data visualization and EDA
* Python tutorial code: [Python_tutorial.ipynb](./Python_tutorial.ipynb)--[colablink](https://colab.research.google.com/drive/1KpLTxgvmFzSlmr486zZwfUBUt-U4-ukT?usp=sharing)
* Python NumPy tutorial code: [Python NumPy tutorial](./Python-Numpy.ipynb)--[colablink](https://colab.research.google.com/drive/10CtxFoyTUk5RIPX4MnOOhYYe3DGAitYW?usp=sharing)
* Data Mining introduction code: 
   * [Dataintro-Pandas.ipynb](./Dataintro-Pandas.ipynb)
   * [Dataintro-EDA.ipynb](./Dataintro-EDA.ipynb)
   * [Dataintro-Visualization.ipynb](./Dataintro-Visualization.ipynb)

Python data apps based on streamlit: [streamlittest](dataapps/streamlittest.py)

## Cloud Data Analytics

* Data Mining based on Google Cloud: 
   * Google Cloud access via Colab: [colablink](https://colab.research.google.com/drive/1fmNMY23wzoQQTGoGns1cgTrjOIuj-Qtc?usp=sharing)
   * Google BigQuery with Colab/Jupyter introduction [BigQuery-intro.ipynb](./BigQuery-intro.ipynb) -- [colablink](https://colab.research.google.com/drive/1HREJs7dUZfrJaPP2wApPNtaINpe2Rtey?usp=sharing)
      * Natality dataset, Weather data
   * COVID19 Data EDA and Visualization based on Google BigQuery (Fall 2022 updated): [colablink](https://colab.research.google.com/drive/1y4zQl_SxA1DEbjI5XjBuxmXQrx5xI1vE?usp=sharing)
      * COVID NYT data, COVID-19 JHU data
   * Additional Google BigQuery examples: [colablink](https://colab.research.google.com/drive/1eHj3g5qwzp4uhE0j0qagCLj5SBWIbuTL?usp=sharing)
      * Chicago Crime Dataset, Austin Waste Dataset, COVID Racial Dataset (race graph)
   * BigQuery ML examples: [colablink](https://colab.research.google.com/drive/1ZX5X9ryN9fq6R1Sb7kEscPaJjRGx0ft3?usp=sharing)
      * COVID, CREDIT_CARD_FRAUD, Predict penguin weight, Natality, US Census Dataset Classification, time-series forecasting from Google Analytics data

## Machine Learning Algorithm
* Machine Learning introduction: 
   * MLIntro-Regression -- [colablink](https://colab.research.google.com/drive/1atrY6rpfPKs5K1VxddfEOWR5QRarxHiG?usp=sharing)
   * MLIntro-RegressionSKLearn -- [colablink](https://colab.research.google.com/drive/1XUzW9vSqyNM02v9F5ueaLMtFb0VorOA3?usp=sharing)
   * [MLIntro2-classification.ipynb](./MLIntro2-classification.ipynb) --[colablink](https://colab.research.google.com/drive/1znfskFZFo-m7VjnI5vgcxdaPWHvLLG9H?usp=sharing)
   * DecisionTree -- [colablink](https://colab.research.google.com/drive/15N_qxOY74batHHjTvkh6zoQ0_85bfdDQ?usp=sharing)
   * GradientBoosting -- [colablink](https://colab.research.google.com/drive/1eT68ZVw3F8Dw1ZjYmfPo3wJutS68S80Q?usp=sharing)
   * XGBoost -- [colablink](https://colab.research.google.com/drive/1ZKtpwoRnK8r2fy8ucXoz1K9E98X76dFC?usp=sharing)

## Deep Learning
Deep learning notebooks
* Tensorflow introduction code: [CMPE-Tensorflow1.ipynb](./CMPE-Tensorflow1.ipynb) -- [colablink](https://colab.research.google.com/drive/188d4pSon4mSAzhGG54zXjWctTOo7Ds53?usp=sharing)
* Pytorch introduction code: [CMPE-pytorch1.ipynb](./CMPE-pytorch1.ipynb) -- [colablink](https://colab.research.google.com/drive/1KZKXqa8FkaJpruUl1XzE7vjvb4pHfMoS?usp=sharing)
* Tensorflow image classification:
   * Road sign data from Kaggle example: [Tensorflow-Roadsignclassification.ipynb](./Tensorflow-Roadsignclassification.ipynb), [colablink](https://colab.research.google.com/drive/1W0bwQVXDFakcB7FdQbbkrSdsucNWW7we)
   * Flower dataset example with TF Dataset, TFRecord, Google Cloud Storage, TPU/GPU acceleration: [colablink](https://colab.research.google.com/drive/1_CwebpyvkcTdAW4zbffga6DT58yw0bZO?usp=sharing)
* Pytorch image classification sample: [CMPE-pytorch2.ipynb](./CMPE-pytorch2.ipynb)
* Advanced Image Classification (Tensorflow and Pytorch): [githubrepo](https://github.com/lkk688/MultiModalClassifier)

New Deep Learning sample code based on Pytorch (under the folder of "DeepDataMiningLearning")
* Pytorch Single GPU image classification: [singleGPU](DeepDataMiningLearning/singleGPU.py)
* Pytorch DDP test: [testTorchDDP](DeepDataMiningLearning/testTorchDDP.py)
* Pytorch Multi-GPU image classification: [multiGPU](DeepDataMiningLearning/multiGPU.py)
* Pytorch ImageNet classification example: [imagenet](DeepDataMiningLearning/imagenet.py)
* Pytorch inference example for top-k class: [inference.py](DeepDataMiningLearning/inference.py)
* TIMM models: [testtimm.ipynb](DeepDataMiningLearning/testtimm.ipynb)
* Huggingface Images via Transformers: [huggingfaceimage.ipynb](DeepDataMiningLearning/huggingfaceimage.ipynb)
* Siamese network: [siamese_network](DeepDataMiningLearning/siamese_network.py)
* TensorRT example: [tensorrt.ipynb](DeepDataMiningLearning/tensorrt.ipynb)
* Object detection (other repo)
   * [MultiModalDetector](https://github.com/lkk688/MultiModalDetector)
   * [myyolov7](https://github.com/lkk688/myyolov7)
   * [WaymoObjectDetection](https://github.com/lkk688/WaymoObjectDetection)
   * [CustomDetectron2](https://github.com/lkk688/CustomDetectron2)

## Unsupervised Learning
* Unsupervised Learning Jupyter notebooks
  * PCA: [colablink](https://colab.research.google.com/drive/1zho_nKQq8yQ-4IFXxw9GZEcdhVdtOabX?usp=share_link)
    * Numpy/SKlearn SVD, PCA for digits and noise filtering, eigenfaces, PCA vs LDA vs NCA
  * Manifold Learning: [colablink](https://colab.research.google.com/drive/1XkCpm7tsnngB7l7AUcrnIo3rKjyfOZev?usp=share_link)
    * Multidimensional Scaling (MDS), Locally Linear Embedding (LLE), Isomap Embedding, T-distributed Stochastic Neighbor Embedding for HELLO, S-Curve, and Swiss roll dataset; Isomap on Faces; Regression with Mainfold Learning
  * Clustering: [colablink](https://colab.research.google.com/drive/1wOMrFR7AXnSc99mUkpJhMLfshpe5aeGd?usp=share_link) 
    * K-Means, Gaussian Mixture Models, Spectral Clustering, DBSCAN 

## NLP and Text Mining
* Text Mining Jupyter notebooks
   * Text Representations: [colablink](https://colab.research.google.com/drive/1L4gyfPqvqdvWSGy88DXVS-7nta1pGWB8?usp=sharing)
      * One-Hot encoding, Bag-of-Words, TF-IDF, and Word2Vec (based on gensim); Word2Vec WiKi and Shakespeare examples; Gather data from Google and WordCLoud
   * Texrtact and NLTK: [colablink](https://colab.research.google.com/drive/1q6Khw3MGJg2S1q8eOcpgtnbLPS_LD7Uj?usp=share_link)
      * Text Extraction via textract; NLTK text preprocessing
   * Text Mining via Tensorflow-text: [colablink](https://colab.research.google.com/drive/1kcM8zAPWDQa1_82OCl74CZOCgZDofipR?usp=share_link)
      * Using Keras embedding layer; sentiment classification example; prepare positive and negative samples and create a Skip-gram Word2Vec model  
   * Text Classification via Tensorflow: [colablink](https://colab.research.google.com/drive/1NyIjdj4d4lueByK-_17BepKLRXz7oM9e?usp=sharing)
      * RNN, LSTM, Transformer, BERT
   * Twitter NLP all-in-one example: [colablink](https://colab.research.google.com/drive/16Lq8pFyxwIUhFi241FYDrG-VfBBSTsgE?usp=sharing)
      * NTLK, LSTM, Bi-LSTM, GRU, BERT

## Recommendation
* Recommendation
   * Recommendation via Python Surprise and Neural Collaborative Filtering (Tensorflow): [colablink](https://colab.research.google.com/drive/1PNi5Vl4YRCsNdLS-pcODSdgbhBlPUoBI?usp=sharing)
   * Tensorflow Recommender: [colab](https://colab.research.google.com/drive/14tfyPInCyZzcr4sk6zRejHR1847WwVR9?usp=sharing)

## Large Language Models (LLMs) and Apps
NLP models based on Huggingface Transformer libraries
* Starting
   * [HuggingfaceTransformers](notebooks/Transformers.ipynb)
   * [huggingfacetest](nlp/huggingfacetest.py)
   * [hfdataset.py](nlp/hfdataset.py)
   * [huggingfaceHPC.ipynb](nlp/huggingfaceHPC.ipynb)
   * [huggingfaceHPCdata](nlp/huggingfaceHPCdata.py)
* Classification application
   * [BERTMTLfakehate](nlp/BERTMTLfakehate.py)
   * [MLTclassifier](nlp/MLTclassifier.py)
   * [huggingfaceClassifierNER.ipynb](nlp/huggingfaceClassifierNER.ipynb)
* Multi-modal Classifier: [huggingfaceclassifier2](nlp/huggingfaceclassifier2.py), [huggingfaceclassifier](nlp/huggingfaceclassifier.py)
* Sequence related application, e.g., translation, summary
   * [huggingfaceSequence](nlp/huggingfaceSequence.ipynb)
* Question and Answer (Q&A)
   * [huggingfaceQA.py](nlp/huggingfaceQA.py)
* Chatbot
   * [huggingfacechatbot.ipynb](nlp/huggingfacechatbot.ipynb)

Pytorch Transformer
* [torchtransformer](nlp/torchtransformer.py)

Open Source LLMs
* [BERTLM.ipynb](nlp/BERTLM.ipynb)
* Masked Language Modeling: [huggingfaceLM.ipynb](nlp/huggingfaceLM.ipynb)
* [llama2](nlp/llama2.ipynb)

LLMs Apps based on OpenAI API
* [openaiqa.ipynb](dataapps/openaiqa.ipynb), [webcrawl.ipynb](dataapps/webcrawl.ipynb)

LLMs Apps based on LangChain
* [langchaintest.ipynb](nlp/langchaintest.ipynb)

