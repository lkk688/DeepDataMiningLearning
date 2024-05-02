# DeepDataMiningLearning
Data mining, machine learning, and deep learning sample codes for SJSU CMPE255 Data Mining ([Fall2023 SJSU Official Syllabus](https://sjsu.campusconcourse.com/view_syllabus?course_id=22871&public_mode=1)) and CMPE258 Deep Learning ([Fall2023 SJSU Official Syllabus](https://sjsu.campusconcourse.com/view_syllabus?course_id=26399&public_mode=1)).
* Some google colab examples need SJSU google account to view)
* Large language Models (LLMs) part is newly added
* You can also view the documents in: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)

## Setups
Install this python package (optional) via

```bash
git clone https://github.com/lkk688/DeepDataMiningLearning.git
cd DeepDataMiningLearning
% pip install flit
% flit install --symlink
```
You will see the package "DeepDataMiningLearning" is install in your python virionment, you can "import DeepDataMiningLearning" as a package.

If you face issues of "OSError: [WinError 1314] A required privilege is not held by the client" in Windows, you can try to activate developer mode on Windows settings (Settings -> System -> For developers (turn on)).

ref "docs/python.rst" for detailed python package description.

Open the Jupyter notebook in local machine:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

## Sphinx docs

Activate python virtual environment, you can use 'sphinx-build' command to build the document

```bash
   % pip install -r requirements.txt
   % pip install nbsphinx, nbsphinx_link  #enable jupyter notebook support for Sphinx: https://docs.readthedocs.io/en/stable/guides/jupyter.html
   % #https://pandoc.org/installing.html
   (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % sphinx-build docs ./docs/build
   #check the integrity of all internal and external links:
   (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % sphinx-build docs -W -b linkcheck -d docs/build/doctrees docs/build/html
```
The generated html files are in the folder of "build". You can also view the documents in: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)

## Python Data Analytics
Basic python tutorials, numpy, Pandas, data visualization and EDA
* Colab basic features: [colabfeatures](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
* Tutorial of Colab working with external data: [colabexternaldata](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=XDg9OBaYqRMd)
* Python tutorial code: [Python_tutorial.ipynb](./Python_tutorial.ipynb)--[colablink](https://colab.research.google.com/drive/1KpLTxgvmFzSlmr486zZwfUBUt-U4-ukT?usp=sharing)
* Python NumPy tutorial code: [Python NumPy tutorial](./Python-Numpy.ipynb)--[colablink](https://colab.research.google.com/drive/10CtxFoyTUk5RIPX4MnOOhYYe3DGAitYW?usp=sharing)
* Data Mining introduction code: 
   * [Dataintro-Pandas.ipynb](./notebooks/Dataintro-Pandas.ipynb) --[colablink](https://colab.research.google.com/drive/14zantNUelF-uPLOXYH8PDzcaPFD-94tc)
   * [Dataintro-EDA.ipynb](./notebooks/Dataintro-EDA.ipynb) --[colablink](https://colab.research.google.com/drive/191Ak-8YzdwJVuCjhFUOJ-WnV0OaDYe2f)
   * [Dataintro-Visualization.ipynb](./notebooks/Dataintro-Visualization.ipynb) --[colablink](https://colab.research.google.com/drive/1zPfz3zma_EriCKvLMShM7jsg5aR_1Cpn)

Python data apps based on streamlit:
* Streamlit tutorial: [streamlittest](dataapps/streamlittest.py)
* Streamlit connect to data sources: [streamlitdata](https://docs.streamlit.io/knowledge-base/tutorials/databases)
* Streamlit connect to Google Big Query: [streamlitbigquery](https://docs.streamlit.io/knowledge-base/tutorials/databases/bigquery)
* Deploy Streamlit to Google Cloud App Engine: [streamlitappengine](https://dev.to/whitphx/how-to-deploy-streamlit-apps-to-google-app-engine-407o)

## Cloud Data Analytics

* Data Mining based on Google Cloud: 
   * Google Cloud access via Colab: [colablink](https://colab.research.google.com/drive/1fmNMY23wzoQQTGoGns1cgTrjOIuj-Qtc?usp=sharing)
      * Configure Gcloud, Google Cloud Storage, Compute Engine, Colab Terminal
   * Google BigQuery with Colab/Jupyter introduction [BigQuery-intro.ipynb](./BigQuery-intro.ipynb) -- [colablink](https://colab.research.google.com/drive/1HREJs7dUZfrJaPP2wApPNtaINpe2Rtey?usp=sharing)
      * BigQuery setup, create and check BigQuery datasets
      * Pandas EDA and visualization based on Natality dataset from BigQuery
      * Weather data from Google BigQuery (revised based on Google's office sample), curve fitting via scipy
   * COVID19 Data EDA and Visualization based on Google BigQuery (Fall 2022 updated): [colablink](https://colab.research.google.com/drive/1y4zQl_SxA1DEbjI5XjBuxmXQrx5xI1vE?usp=sharing)
      * COVID NYT data from BigQuery: prediction of CA cases via fbprophet, states with hightest cases, cases (moving average) over time curve, Heatmap of Confirmed Cases, Statewise Mask usage habits, joint data of Population from Census data, zipcode data, and impact of mask usage. 
      * COVID-19 JHU data: case map view, top10 states, moving average, ARIMA model, save dataframe back to BigQuery
   * Additional Google BigQuery examples: [colablink](https://colab.research.google.com/drive/1eHj3g5qwzp4uhE0j0qagCLj5SBWIbuTL?usp=sharing)
      * Chicago Crime Dataset, Austin Waste Dataset, COVID Racial Dataset (race graph)
   * Student Samples
      * [SFTheft dataset](https://colab.research.google.com/drive/1FC7GoFHuAClmw8GTm2bq_yxKA4qOuWbi?usp=sharing)
      * [AirBnB dataset](https://colab.research.google.com/drive/1xiXWBFsxw2v86ACGGCLx-aB5M6B0Zb-r?usp=sharing)
   * BigQuery ML examples: [colablink](https://colab.research.google.com/drive/1ZX5X9ryN9fq6R1Sb7kEscPaJjRGx0ft3?usp=sharing)
      * COVID, CREDIT_CARD_FRAUD, Predict penguin weight, Natality, US Census Dataset Classification, time-series forecasting from Google Analytics data
   * BigQuery Bigframe and ML examples (2024 update): [colablink](https://colab.research.google.com/drive/1Qv9EhSUQpQsBXZsvxFg_IyeeryepQ_hE?usp=sharing)
      * BigQuery DataFrames: bigframes
      * Bigquery ML for supervised and unsupervised learning with SKLearn stype API
      * Bigquery LLM for text generation, code generation (pandas api code)
      * Text Embedding, Kmeans for text embeddings
      * Use PaLM2 LLM model to summarize text/complaints

## Machine Learning Algorithm
* Machine Learning introduction: 
   * MLIntro-Regression -- [colablink](https://colab.research.google.com/drive/1atrY6rpfPKs5K1VxddfEOWR5QRarxHiG?usp=sharing)
      * Linear Regression via Normal Equation and own SGD implementation
      * Diamond dataset from Kaggle, EDA, and predict the price of diamond via normal equation, numpy.linalg
   * MLIntro-RegressionSKLearn -- [colablink](https://colab.research.google.com/drive/1XUzW9vSqyNM02v9F5ueaLMtFb0VorOA3?usp=sharing)
      * Simulation data: linear regression, polynomial regression, Ridge and Lasso regression
      * Diamond dataset: Categorical Feature Encoding, Linear Regression and other regressors
      * Boston housing dataset: Regression Model for All the variables, Ridge and Lasso Regression
      * Diabetes Dataset: Linear regression, Ridge and Lasso Regression
      * AUTO MPG Dataset: 1) Linear regression; 2) Polynomial Regression via SKlearn PolynomialFeatures; 3) Multiple Linear Regression; 4) SKLearn Pipelines combine feature engineering stages and the model into a single "model"; 5) ColumnTransformer and FunctionTransformer; 6) Cross Validation
   * [MLIntro2-classification.ipynb](./MLIntro2-classification.ipynb) --[colablink](https://colab.research.google.com/drive/1znfskFZFo-m7VjnI5vgcxdaPWHvLLG9H?usp=sharing)
      * Breast Cancer Dataset, iris Dataset, BigQuery US Census Income Dataset, multiple classifiers. 
   * DecisionTree -- [colablink](https://colab.research.google.com/drive/15N_qxOY74batHHjTvkh6zoQ0_85bfdDQ?usp=sharing)
      * SKlearn DecisionTree algorithm on Iris dataset, Breast Cancel Dataset, Make moon dataset, and DecisionTreeRegressor. A berif discussion of Gini Impurity.    
   * GradientBoosting -- [colablink](https://colab.research.google.com/drive/1eT68ZVw3F8Dw1ZjYmfPo3wJutS68S80Q?usp=sharing)
      *  Gradient boosting process, Gradient boosting regressor with scikit-learn, Gradient boosting classifier with scikit-learn
   * XGBoost -- [colablink](https://colab.research.google.com/drive/1ZKtpwoRnK8r2fy8ucXoz1K9E98X76dFC?usp=sharing)
      * XGBoost introduction, US Census Income Dataset from Big Query, UCI Dermatology dataset
   * Naive Bayes Classifier -- [colablink](https://colab.research.google.com/drive/1a0e6XkaeMZiqy-NyRa6uX_jAyGa26u2e?usp=sharing)
      * Bayesian Classification, Naive Bayes in SKlearn, Gaussian Naive Bayes, Multinomial Naive Bayes Classifying Text (20 Newsgroups)
      * Naive Bayes for MNIST from Torchvision
      * NVIDIA RAPIDS installation, NVIDIA Cuml Naive Bayes vs Sklearn Naive Bayes based on News Aggregator dataset
   * Support Vector Machine -- [colablink](https://colab.research.google.com/drive/1ZHBUREOF_midVVLe46Baot4HHdil89SU?usp=sharing)
      * Support Vector Machines introduction, SVM for Breast Cancer Dataset, SVM for Face Recognition
      * SVM for MNIST in Pytorch, Neural Network MNIST Pytorch

## Deep Learning
Deep learning notebooks (colab link is better)
* Tensorflow introduction code: [CMPE-Tensorflow1.ipynb](./notebooks/CMPE-Tensorflow1.ipynb) -- [colablink](https://colab.research.google.com/drive/188d4pSon4mSAzhGG54zXjWctTOo7Ds53?usp=sharing)
* Pytorch introduction code: 
   * Pytorch Introductions, Tensors, and Autograd: [colablink](https://colab.research.google.com/drive/1_B7F59i51IlQoZPy_v85ux2WRucIRfWo?usp=sharing)
   * Pytorch Regression and Logistic Regression: [colablink](https://colab.research.google.com/drive/17KgzFzRx6BwsEC6iATLSqOczT_EwcSlb?usp=sharing)
   * Pytorch Simple Neural Networks: [colab](https://colab.research.google.com/drive/1HeZT2QphKO5c2gTtE6_TqYXfDzxcsJGc?usp=sharing)
   * Pytorch NN modules: [pytorch_nn.ipynb](notebooks/pytorch_nn.ipynb)
* Tensorflow image classification:
   * Road sign data from Kaggle example: [Tensorflow-Roadsignclassification.ipynb](./notebooks/Tensorflow-Roadsignclassification.ipynb), [colablink](https://colab.research.google.com/drive/1W0bwQVXDFakcB7FdQbbkrSdsucNWW7we)
   * Flower dataset example with TF Dataset, TFRecord, Google Cloud Storage, TPU/GPU acceleration: [colablink](https://colab.research.google.com/drive/1_CwebpyvkcTdAW4zbffga6DT58yw0bZO?usp=sharing)
* Pytorch image classification:
   * Pytorch image classification introduction (MNIST, CNN filters, CIFAR, VGGNet, Flowers): [colablink](https://colab.research.google.com/drive/19Lhu8vM-3PwDc-suNhFvIrmTM-ji-qwe?usp=sharing)
   * Pytorch Single GPU image classification with/without automatic mixed precision (AMP) training: [singleGPU](DeepDataMiningLearning/singleGPU.py)
   * Pytorch Multi-GPU image classification: [multiGPU](DeepDataMiningLearning/multiGPU.py)
   * Pytorch Torchvision image classification (Efficientnet) notebook on HPC: [torchvisionHPC.ipynb](DeepDataMiningLearning/torchvisionHPC.ipynb)
   * Pytorch Torchvision vision transformer (ViT) notebook on HPC: [torchvisionvitHPC.ipynb](DeepDataMiningLearning/torchvisionvitHPC.ipynb)
   * Pytorch ImageNet classification example: [imagenet](DeepDataMiningLearning/imagenet.py)
   * Pytorch inference example for top-k class: [inference.py](DeepDataMiningLearning/inference.py)
   * TIMM models: [testtimm.ipynb](DeepDataMiningLearning/testtimm.ipynb)
   * Huggingface Transformers for Image: [hfvisionmain.py](DeepDataMiningLearning/hfvisionmain.py)
* Advanced Multi-Modal Image Classification: [githubrepo](https://github.com/lkk688/MultiModalClassifier)
   * General purpose framework for all-in-one image classification for Tensorflow and Pytorch
   * Support for multiple datasets: imagenet_blurred, tiny-imagenet-200, hymenoptera_data, CIFAR10, MNIST, flower_photos
   * Support for multiple custom models ('mlpmodel1', 'lenet', 'alexnet', 'resnetmodel1', 'customresnet', 'vggmodel1', 'vggcustom', 'cnnmodel1'), all models from Torchvision and TorchHub
   * Support HPC training and evaluation
  
New Deep Learning sample code based on Pytorch (under the folder of "DeepDataMiningLearning")
* Pytorch Multi-GPU DDP test: [testTorchDDP](DeepDataMiningLearning/testTorchDDP.py)
* Pytorch ViT implement from scratch on HPC: [ViTHPC.ipynb](DeepDataMiningLearning/ViTHPC.ipynb)
* Huggingface Images via Transformers: [huggingfaceimage.ipynb](DeepDataMiningLearning/huggingfaceimage.ipynb)
* Siamese network: [siamese_network](DeepDataMiningLearning/siamese_network.py)
* TensorRT example: [tensorrt.ipynb](DeepDataMiningLearning/tensorrt.ipynb)
* Object detection (other repo)
   * [MultiModalDetector](https://github.com/lkk688/MultiModalDetector)
   * [myyolov7](https://github.com/lkk688/myyolov7): Add YOLOv5 models with YOLOv7, performed training on COCO and WaymoCOCO dataset.
   * [myyolov5](https://github.com/lkk688/yolov5): My fork of the YOLOv5, convert COCO to YOLO format, changed the code to be the base code for YOLOv4, YOLOv5, and ScaledYOLOv4; performed training on COCO and WaymoCOCO dataset.
   * [WaymoObjectDetection](https://github.com/lkk688/WaymoObjectDetection)
      * Waymo Dataset Conversion to COCO format: WaymoCOCO
      * [torchvision_waymococo_train.py](https://github.com/lkk688/WaymoObjectDetection/blob/master/MyDetector/torchvision_waymococo_train.py): performs Pytorch FasterRCNN training based on converted Waymo COCO format data. This version can be applied for any dataset with COCO format annotation
      * [WaymoCOCODetectron2train.py](https://github.com/lkk688/WaymoObjectDetection/blob/master/2DObject/WaymoCOCODetectron2train.py): WaymoCOCO training based on Detectron2
      * [mymmdetection2dtrain.py](https://github.com/lkk688/WaymoObjectDetection/blob/master/2DObject/mymmdetection2dtrain.py): Object Detection training and evaluation based on MMdetection2D
   * [CustomDetectron2](https://github.com/lkk688/CustomDetectron2)

## Unsupervised Learning
* Unsupervised Learning Jupyter notebooks
  * PCA: [colablink](https://colab.research.google.com/drive/1zho_nKQq8yQ-4IFXxw9GZEcdhVdtOabX?usp=share_link)
    * Affine Transformation via Matrix Application, eigenvalues and eigenvectors, eigendecomposition
    * Numpy/SKlearn SVD, SVD for linear regression
    * Principal Component Analysis in SKLearn, PCA as dimensionality reduction, PCA for SkLearn Iris dataset
    * PCA for Image Compression, PCA for digits and noise filtering, Grey Image Example, Color Image Example, eigenfaces, PCA vs LDA vs NCA
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

## NLP Models and Transformer
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


## Large Language Models (LLMs)
Train a basic language modeling task via basic Pytorch and Torchtext WikiText2 dataset in HPC. 
```bash
python nlp/torchtransformer.py

| epoch   1 |  2800/ 2928 batches | lr 5.00 | ms/batch  5.58 | loss  3.31 | ppl    27.49
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 24.00s | valid loss  1.96 | valid ppl     7.08
-----------------------------------------------------------------------------------------
| epoch   2 |   200/ 2928 batches | lr 4.75 | ms/batch  5.84 | loss  3.07 | ppl    21.57
| epoch   2 |  2800/ 2928 batches | lr 4.75 | ms/batch  5.49 | loss  2.58 | ppl    13.26
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 1655.94s | valid loss  1.52 | valid ppl     4.57
-----------------------------------------------------------------------------------------
| epoch   3 |   200/ 2928 batches | lr 4.51 | ms/batch  5.04 | loss  2.41 | ppl    11.15
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 15.41s | valid loss  1.44 | valid ppl     4.22
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  1.40 | test ppl     4.06
=========================================================================================
```

Train Masked Language model:
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceLM2.py --data_name="eli5" --model_checkpoint="distilroberta-base" --task="CLM" --subset=5000 --traintag="1115CLM" --usehpc=True --gpuid=1 --batch_size=32 --learningrate=2e-5

python nlp/huggingfaceLM2.py
data_type=huggingface data_name=eli5 dataconfig= subset=0 data_path=/data/cmpe249-fa23/Huggingfacecache model_checkpoint=distilroberta-base task=MLM unfreezename= outputdir=./output traintag=1116MLM training=True usehpc=False gpuid=0 total_epochs=8 save_every=2 batch_size=32 learningrate=2e-05
Trainoutput folder: ./output\distilroberta-base\eli5_1116MLM
....
Epoch 1: Perplexity: 12.102828644322578                              | 6590/26360 [16:06:09<34:50:06,  6.34s/it]
 38%|███████████████████████>>> Epoch 2: Perplexity: 15.187707787848385                              | 9885/26360 [24:00:07<28:57:25,  6.33s/it]
 50%|███████████████████████>>> Epoch 3: Perplexity: 15.063201196763071                             | 13180/26360 [31:52:08<23:12:51,  6.34s/it]
 62%|███████████████████████>>> Epoch 4: Perplexity: 16.583895970053355                             | 16475/26360 [39:44:32<17:23:28,  6.33s/it]
 75%|███████████████████████>>> Epoch 5: Perplexity: 16.27479412837067██████▎                       | 19770/26360 [47:36:46<11:34:43,  6.33s/it]
 88%|███████████████████████>>> Epoch 6: Perplexity: 16.424729093343636██████████████████            | 23065/26360 [55:28:38<5:47:18,  6.32s/it]
100%|███████████████████████>>> Epoch 7: Perplexity: 17.22636450834783
```


Train GPT2 language models
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceLM2.py --model_checkpoint="gpt2" --task="CLM" --traintag="1115gpt2" --usehpc=True --gpuid=2 --batch_size=16
```
Train llama2 7b model and only unfreeze the last layers "model.layers.31" (need 500GB) or "lm_head" (need 40GB)
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceLM2.py --model_checkpoint="Llama-2-7b-chat-hf" --task="CLM" --unfreezename="lm_head" --traintag="1115llama2" --usehpc=True --gpuid=2 --batch_size=8

python nlp/huggingfaceLM2.py --model_checkpoint="Llama-2-7b-chat-hf" --pretrained=="/data/cmpe249-fa23/trainoutput/huggingface/Llama-2-7b-chat-hf/eli5_1115llama2/savedmodel.pth" --task="CLM" --unfreezename="lm_head" --traintag="1119llama2" --usehpc=True --gpuid=0 --batch_size=8
.....
Epoch 0: Perplexity: 9.858825392857694████████████████████████████████████████| 2627/2627 [12:39:17<00:00,  3.30s/it]
 25%|█████████████████▊     >>> Epoch 1: Perplexity: 10.051054027867561     | 21014/84056 [22:50:31<56:09:49,  3.21s/it]
 38%|███████████████████████>>> Epoch 2: Perplexity: 10.181400762228291     | 31521/84056

Epoch 0: Perplexity: 9.289763256151375
Epoch 1: Perplexity: 9.530650993830372
Epoch 2: Perplexity: 9.692566051540275

```

## LLMs for Translation
Train translation models based on huggingfaceSequence
```bash
python nlp/huggingfaceSequence.py --data_name="kde4" --model_checkpoint="Helsinki-NLP/opus-mt-en-fr" --task="Seq2SeqLM" --traintag="1116" --usehpc=True --gpuid=0 --batch_size=8

epoch 0, BLEU score: 51.78█████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [21:18<00:00,  3.00it/s] 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [22:54<00:00,  1.91it/s]
epoch 1, BLEU score: 52.73█████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [22:54<00:00,  2.81it/s] 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [23:06<00:00,  1.90it/s]
epoch 2, BLEU score: 54.
epoch 3, BLEU score: 54.
epoch 4, BLEU score: 55.
epoch 5, BLEU score: 55.
epoch 6, BLEU score: 54.
epoch 7, BLEU score: 55.
```

```bash
python nlp/huggingfaceSequence.py --data_name="opus100" --model_checkpoint="facebook/wmt21-dense-24-wide-en-x" --task="Seq2SeqLM" --traintag="1121" --usehpc=True --gpuid=1 --batch_size=8
```

```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceSequence2.py --data_name="opus100" --subset=0 --model_checkpoint="Helsinki-NLP/opus-mt-en-zh" --task="Seq2SeqLM" --traintag="1122" --evaluate="" --usehpc=True --
gpuid=1 --batch_size=32

python nlp/huggingfaceSequence2.py --data_name="opus100" --subset=0 --model_checkpoint="Helsinki-NLP/opus-mt-en-zh" --pretrained="/data/cmpe249-fa23/trainoutput/huggingface/Helsinki-NLP/opus-mt-en-zh/opus100_1122/savedmodel.pth" --task="Seq2SeqLM" --target_lang="zh" --traintag="1122" --evaluate=True --usehpc=True --gpuid=1 --total_epochs=16 --batch_size=32
.....
epoch 14, BLEU score: 48.55
epoch 15, BLEU score: 48.53

python nlp/huggingfaceSequence2.py --data_name="wmt19" --subset=0 --model_checkpoint="Helsinki-NLP/opus-mt-en-zh" --task="Seq2SeqLM" --target_lang="zh" --traintag="1123" --evaluate="localevaluate" --usehpc=True --gpuid=2 --total_epochs=16 --batch_size=32

(mycondapy310) [010796032@cs002 DeepDataMiningLearning]$ python nlp/huggingfaceSequence2.py --data_name="wmt19" --subset=10000 --model_checkpoint="t5-base" --task="Seq2SeqLM" --target_lang="zh" --traintag="1123" --useHFaccelerator=True --ev
aluate="localevaluate" --usehpc=True --gpuid=2 --total_epochs=16 --batch_size=64
.....
Trainoutput folder: /data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1123
epoch 0, BLEU score: 2.66
epoch 1, BLEU score: 3.99
epoch 2, BLEU score: 5.31
epoch 3, BLEU score: 6.64
epoch 4, BLEU score: 8.07
epoch 5, BLEU score: 9.51
epoch 9, BLEU score: 14.39
epoch 14, BLEU score: 18.99
epoch 15, BLEU score: 19.77

(mycondapy310) [010796032@cs002 DeepDataMiningLearning]$ python nlp/huggingfaceSequence2.py --data_name="wmt19" --subset=50000 --model_checkpoint="t5-base" --task="Seq2SeqLM" --target_lang="zh" --traintag="1124" --pretrained="/data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1123/savedmodel.pth" --useHFaccelerator=True --evaluate="localevaluate" --usehpc=True --gpuid=2 --total_epochs=32 --batch_size=64
epoch 16, BLEU score: 50.83
epoch 31, BLEU score: 56.57

python nlp/huggingfaceSequence2.py --data_name="wmt19" --subset=50000 --model_checkpoint="t5-base" --task="Seq2SeqLM" --target_lang="zh" --source_prefix="translate English to Chinese: " --traintag="1124" --pretrained="/data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1124/savedmodel.pth" --useHFaccelerator=True --evaluate="localevaluate" --usehpc=True --total_epochs=48 --batch_size=64
Trainoutput folder: /data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1124
epoch 32, BLEU score: 9.67
epoch 33, BLEU score: 13.95
epoch 35, BLEU score: 20.78
epoch 45, BLEU score: 37.60
epoch 47, BLEU score: 39.47

python nlp/huggingfaceSequence2.py --data_name="wmt19" --subset=50000 --model_checkpoint="t5-base" --task="Seq2SeqLM" --target_lang="zh" --source_prefix="translate English to Chinese: " --traintag="1124" --pretrained="/data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1124/savedmodel.pth" --useHFaccelerator=True --evaluate="localevaluate" --usehpc=True --total_epochs=64 --batch_size=64
epoch 48, BLEU score: 60.04
epoch 63, BLEU score: 59.26

$ python nlp/huggingfaceSequence2.py --data_name="wmt19" --subset=0 --model_checkpoint="t5-base" --task="Seq2SeqLM" --target_lang="zh" --source_prefix="translate English to Chinese: " --traintag="1124" --pretrained="/data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1124/savedmodel.pth" --useHFaccelerator=False --evaluate="localevaluate" --usehpc=True --gpuid=2 --total_epochs=80 --batch_size=64
```

Train T5-base in local computer
```bash
/nlp/huggingfaceSequence2.py
data_type=huggingface data_name=opus100 dataconfig= subset=0 data_path=/data/cmpe249-fa23/Huggingfacecache model_checkpoint=t5-base task=Seq2SeqLM evaluate=True source_lang=en target_lang=zh source_prefix=None pretrained= unfreezename= outputdir=./output traintag=1122 training=True usehpc=False useHFaccelerator=False gpuid=0 total_epochs=8 save_every=2 batch_size=16 learningrate=2e-05 lr_scheduler_type=linear weight_decay=0.0 gradient_accumulation_steps=1 pad_to_max_length=True max_source_length=128 max_target_length=128 num_beams=1
Trainoutput folder: ./output\t5-base\opus100_1122
epoch 0, BLEU score: 48.70
epoch 1, BLEU score: 50.81
epoch 2, BLEU score: 50.24
epoch 3, BLEU score: 51.93
epoch 4, BLEU score: 52.34
epoch 5, BLEU score: 52.54
epoch 6, BLEU score: 52.71
epoch 7, BLEU score: 52.91
(mycondapy39) PS C:\Users\lkk68\Documents\GitHub\DeepDataMiningLearning> cat .\output\t5-base\opus100_1122\eval_results.json
   {"eval_bleu": 52.909234849408264}

nlp/huggingfaceSequence2.py
data_type=huggingface data_name=opus_books dataconfig= subset=0 data_path=/data/cmpe249-fa23/Huggingfacecache model_checkpoint=t5-base task=Seq2SeqLM evaluate=localevaluate source_lang=en target_lang=fr source_prefix=None pretrained= unfreezename= outputdir=./output traintag=1124 training=True usehpc=False useHFaccelerator=False gpuid=0 total_epochs=16 save_every=2 batch_size=16 learningrate=2e-05 lr_scheduler_type=linear weight_decay=0.0 gradient_accumulation_steps=1 pad_to_max_length=True max_source_length=128 max_target_length=128 num_beams=1
Trainoutput folder: ./output\t5-base\opus_books_1124
HF evaluator: 24.46
epoch 0, BLEU score: 24.47
HF evaluator: 26.00
epoch 14, BLEU score: 26.00
HF evaluator: 25.89
epoch 15, BLEU score: 25.90
```

## LLMs for Summarization
Train summarization model based on "cnn_dailymail" dataset
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceSequence3.py --data_name="cnn_dailymail" --subset=0 --model_checkpoint="t5-base" --training --usehpc --task="summarization" --source_prefix="summarize: " --traintag="1125" --gpuid=1 --total_epochs=8 --batch_size=32
useHFevaluator: False
dualevaluator: False
data_type=huggingface data_name=cnn_dailymail dataconfig= subset=0.0 data_path=/data/cmpe249-fa23/Huggingfacecache model_checkpoint=t5-base task=summarization hfevaluate=False dualevaluate=False source_lang=en target_lang=fr source_prefix=summarize:  pretrained= unfreezename= outputdir=./output traintag=1125 training=True usehpc=True useHFaccelerator=False gpuid=1 total_epochs=8 save_every=2 batch_size=32 learningrate=2e-05 lr_scheduler_type=linear weight_decay=0.0 gradient_accumulation_steps=1 pad_to_max_length=True max_source_length=128 max_target_length=128 num_beams=1
Trainoutput folder: /data/cmpe249-fa23/trainoutput/huggingface/t5-base/cnn_dailymail_1125

epoch 0, evaluation metric: rouge
Evaluation result: {'rouge1': AggregateScore(low=Score(precision=0.41197173103605056, recall=0.3119407931701639, fmeasure=0.3419831177812576), mid=Score(precision=0.41485220723500893, recall=0.3142588821867073, fmeasure=0.3441474199740058), high=Score(precision=0.417600698303463, recall=0.3165054333064982, fmeasure=0.3464475097686251)), 'rouge2': AggregateScore(low=Score(precision=0.17079493326536863, recall=0.12954017819776567, fmeasure=0.14168846158539045), mid=Score(precision=0.1732732332588367, recall=0.13143856889123576, fmeasure=0.14365393002489873), high=Score(precision=0.17556212138667857, recall=0.1333017602307039, fmeasure=0.14553710285584648)), 'rougeL': AggregateScore(low=Score(precision=0.2965678275511713, recall=0.22464618504694173, fmeasure=0.24595299928001602), mid=Score(precision=0.2989426482943214, recall=0.22661167405971072, fmeasure=0.24784869235990342), high=Score(precision=0.30157116940828527, recall=0.22857528928797705, fmeasure=0.24990879681324848)), 'rougeLsum': AggregateScore(low=Score(precision=0.29642059763679396, recall=0.22452637387862667, fmeasure=0.2458545440066565), mid=Score(precision=0.298870626902608, recall=0.22656017215162805, fmeasure=0.24780842037777753), high=Score(precision=0.3013588263275077, recall=0.22874002983645225, fmeasure=0.24986052268174044))}
.....
epoch 7, evaluation metric: rouge
Evaluation result: {'rouge1': AggregateScore(low=Score(precision=0.4106620116159846, recall=0.3359116786022911, fmeasure=0.3571395048710354), mid=Score(precision=0.4113719796401594, recall=0.3365233125614775, fmeasure=0.3577153054159934), high=Score(precision=0.41210309679230933, recall=0.3371437303658093, fmeasure=0.3583162407770864)), 'rouge2': AggregateScore(low=Score(precision=0.17094983781529516, recall=0.140263702081492, fmeasure=0.1487497359572444), mid=Score(precision=0.17152532221996442, recall=0.14077116290613728, fmeasure=0.1492659697570951), high=Score(precision=0.17217316198104332, recall=0.14133598283522728, fmeasure=0.14982901584825192)), 'rougeL': AggregateScore(low=Score(precision=0.29437670375737585, recall=0.241532114907138, fmeasure=0.2562224943410107), mid=Score(precision=0.295004497912973, recall=0.24202153711108693, fmeasure=0.256706856828388), high=Score(precision=0.29558396252033226, recall=0.2425496385594932, fmeasure=0.25720842817590284)), 'rougeLsum': AggregateScore(low=Score(precision=0.2943661396563572, recall=0.24149928244505112, fmeasure=0.2561939698587155), mid=Score(precision=0.29501206428652565, recall=0.24205477312718807, fmeasure=0.25673184483277667), high=Score(precision=0.2956055704727816, recall=0.2425905810654754, fmeasure=0.25723630538221925))}
```

Train summarization based on "billsum" dataset
```bash
python nlp/huggingfaceSequence3.py --data_name="billsum" --subset=0 --model_checkpoint="t5-base" --training --usehpc --task="summarization" --source_prefix="summarize: " --traintag="1125" --gpuid=2 --total_epochs=8 --batch_size=64
epoch 0, evaluation metric: rouge
Evaluation result: {'rouge1': AggregateScore(low=Score(precision=0.47626942983509496, recall=0.25432053054933895, fmeasure=0.3104459523870372), mid=Score(precision=0.48118215557348354, recall=0.25758086912668254, fmeasure=0.31351244843801257), high=Score(precision=0.48633004345160047, recall=0.2610377756856256, fmeasure=0.31666293379599264)), 'rouge2': AggregateScore(low=Score(precision=0.22553581027732506, recall=0.1163356040224796, fmeasure=0.1426990222648773), mid=Score(precision=0.22980182467297072, recall=0.11916438705516172, fmeasure=0.14559973720411068), high=Score(precision=0.23448683629864495, recall=0.12186100986547159, fmeasure=0.14852517085299424)), 'rougeL': AggregateScore(low=Score(precision=0.37475945811360656, recall=0.20056913183887007, fmeasure=0.24410014882029277), mid=Score(precision=0.3791703619589465, recall=0.2035902141913645, fmeasure=0.24684120728025757), high=Score(precision=0.38361974259418913, recall=0.20659199136456374, fmeasure=0.24957227859403597)), 'rougeLsum': AggregateScore(low=Score(precision=0.3748793629512802, recall=0.200607724566369, fmeasure=0.24403997308680897), mid=Score(precision=0.379214986503034, recall=0.20357105315951063, fmeasure=0.24678602914830827), high=Score(precision=0.38368049681408073, recall=0.20677635875973946, fmeasure=0.2497749989951042))}
.....
epoch 7, evaluation metric: rouge
Evaluation result: {'rouge1': AggregateScore(low=Score(precision=0.4762322399523528, recall=0.3612750168693367, fmeasure=0.3859539337308937), mid=Score(precision=0.47741449894622395, recall=0.36223979353915375, fmeasure=0.3867387680371833), high=Score(precision=0.47857623413135664, recall=0.36315881475914685, fmeasure=0.387529008724504)), 'rouge2': AggregateScore(low=Score(precision=0.23795846378692775, recall=0.17739298212614543, fmeasure=0.1895061898823578), mid=Score(precision=0.23901236113717203, recall=0.17821036280792862, fmeasure=0.1902598464683951), high=Score(precision=0.24012950529243016, recall=0.1789825350527308, fmeasure=0.19103010704737328)), 'rougeL': AggregateScore(low=Score(precision=0.38008605052931843, recall=0.28951287826449035, fmeasure=0.307987126768515), mid=Score(precision=0.3811360590765148, recall=0.29038102311603803, fmeasure=0.30870601375792006), high=Score(precision=0.38221337267972383, recall=0.2911927830141367, fmeasure=0.3094361849338897)), 'rougeLsum': AggregateScore(low=Score(precision=0.3800857948090643, recall=0.2895777984554257, fmeasure=0.3080188262526704), mid=Score(precision=0.38110308526126024, recall=0.2903936893613658, fmeasure=0.3087163027737505), high=Score(precision=0.3821455084830249, recall=0.29123147942293837, fmeasure=0.3094164695199439))}
```
Train summarization based on "xsum" dataset
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceSequence3.py --data_name="xsum" --subset=
0 --model_checkpoint="t5-base" --training --usehpc --task="summarization" --source_prefix="summarize: " --traintag="1125
" --gpuid=1 --total_epochs=8 --batch_size=64
epoch 0, evaluation metric: rouge
Evaluation result: {'rouge1': AggregateScore(low=Score(precision=0.2206634425318806, recall=0.27398080027290334, fmeasure=0.23345078599469463), mid=Score(precision=0.2228479631573893, recall=0.2758314532199041, fmeasure=0.2352387063491203), high=Score(precision=0.22497474810005727, recall=0.27769139479498234, fmeasure=0.23719704189066906)), 'rouge2': AggregateScore(low=Score(precision=0.04855549563663333, recall=0.053939101970654914, fmeasure=0.04875180161702199), mid=Score(precision=0.049879370509959026, recall=0.05516244414389607, fmeasure=0.04995842715629054), high=Score(precision=0.05118590600673647, recall=0.05640622198150646, fmeasure=0.05111651526817903)), 'rougeL': AggregateScore(low=Score(precision=0.1618905690819521, recall=0.19780661742397618, fmeasure=0.16987613018217562), mid=Score(precision=0.16369315145103713, recall=0.19940773965682818, fmeasure=0.17143884336484203), high=Score(precision=0.16554847486855678, recall=0.20099105646119897, fmeasure=0.17303926810412723)), 'rougeLsum': AggregateScore(low=Score(precision=0.16171036371143727, recall=0.19775741367206912, fmeasure=0.16972996097941762), mid=Score(precision=0.16364819272318623, recall=0.19936460332720274, fmeasure=0.1713745081258375), high=Score(precision=0.16540503402328743, recall=0.20093144657877907, fmeasure=0.17297056660820934))}
....
epoch 7, evaluation metric: rouge
Evaluation result: {'rouge1': AggregateScore(low=Score(precision=0.3157722202905392, recall=0.30514034915305777, fmeasure=0.30098661341663313), mid=Score(precision=0.3164460427364083, recall=0.30573870098459877, fmeasure=0.3015941401174891), high=Score(precision=0.31710221504502506, recall=0.30627353921164463, fmeasure=0.3021231398386561)), 'rouge2': AggregateScore(low=Score(precision=0.09545526361853192, recall=0.08805500161867533, fmeasure=0.08929094188051703), mid=Score(precision=0.0959138137320461, recall=0.08846040412480724, fmeasure=0.08970127642926173), high=Score(precision=0.09638196118158575, recall=0.08887820756655522, fmeasure=0.09011808402709783)), 'rougeL': AggregateScore(low=Score(precision=0.2439907968202854, recall=0.23420304190160135, fmeasure=0.23192429294508443), mid=Score(precision=0.2445882204337065, recall=0.23468338778240327, fmeasure=0.23243230200308637), high=Score(precision=0.2451722573264371, recall=0.23517123971453874, fmeasure=0.2329239687124055)), 'rougeLsum': AggregateScore(low=Score(precision=0.24400824561132867, recall=0.23421060757686038, fmeasure=0.2319498933428253), mid=Score(precision=0.24459781572922448, recall=0.23469061202638752, fmeasure=0.2324341602958753), high=Score(precision=0.24514334749025868, recall=0.23516229230273153, fmeasure=0.23293317648718587))}
```
Run question and answering for squad dataset based on huggingfaceSequence4.py.
```bash
nlp/huggingfaceSequence4.py

HF evaluator: {'exact_match': 0.22, 'f1': 6.222554522104021}
Start training, total steps: 79696
epoch 0, evaluation metric: squad
Evaluation result: {'exact_match': 63.36, 'f1': 77.10714274394753}
epoch 15, evaluation metric: squad
Evaluation result: {'exact_match': 62.08, 'f1': 75.95170387159816}
```

## LLMs for Question and Answering
Run question and answering for squad dataset based on custom bert model in huggingfaceSequence4.py.
```bash
nlp/huggingfaceSequence4.py
epoch 0: {'exact_match': 0.7663197729422895, 'f1': 0.8230842005676446}
epoch 1: {'exact_match': 0.7947019867549668, 'f1': 0.8360138757489753}
epoch 5: {'exact_match': 0.8609271523178808, 'f1': 0.8607573442010529}
epoch 7: {'exact_match': 0.8703878902554399, 'f1': 0.8735414695679595}
```

Run open-ended question and answering for squadv2 dataset based on T5 model in huggingfaceSequence5.py
```bash
epoch 0, evaluation metric: squad_v2
Evaluation result: {'exact': 75.42, 'f1': 81.2122560694999, 'total': 5000, 'HasAns_exact': 73.17148125384142, 'HasAns_f1': 82.07169033420416, 'HasAns_total': 3254, 'NoAns_exact': 79.61053837342497, 'NoAns_f1': 79.61053837342497, 'NoAns_total': 1746, 'best_exact': 75.36, 'best_exact_thresh': 0.0, 'best_f1': 81.15225606949971, 'best_f1_thresh': 0.0}
......
epoch 15, evaluation metric: squad_v2
Evaluation result: {'exact': 76.38, 'f1': 82.8391234679352, 'total': 5000, 'HasAns_exact': 73.26367547633681, 'HasAns_f1': 83.18857324513708, 'HasAns_total': 3254, 'NoAns_exact': 82.18785796105384, 'NoAns_f1': 82.18785796105384, 'NoAns_total': 1746, 'best_exact': 76.32, 'best_exact_thresh': 0.0, 'best_f1': 82.77912346793502, 'best_f1_thresh': 0.0}
```

## Multi-task LLMs
Use Sequence5.py for translation training
```bash
python nlp/huggingfaceSequence5.py --data_name="wmt19" --subset=100000 --model_checkpoint="t5-base" --task="translation" --target_lang="zh" --source_prefix="translate English to Chinese: " --traintag="1206" --pretrained="/data/cmpe249-fa23/trainoutput/huggingface/t5-base/wmt19_1124/savedmodel.pth" --usehpc --gpuid=1 --total_epochs=80 --batch_size=64
```

Fine tune "liam168/trans-opus-mt-en-zh" on wmt19 5000 subset
```bash
epoch 15, evaluation metric: sacrebleu
Evaluation result: {'score': 45.11294317735652, 'counts': [200677, 140858, 104286, 81486], 'totals': [283438, 274938, 266464, 258142], 'precisions': [70.8010217402042, 51.232641541001975, 39.1369941155278, 31.566347204251922], 'bp': 0.9805091347328498, 'sys_len': 283438, 'ref_len': 289017}
```

## SignalAI
Perform audio classification via "hfclassify1.py":
```bash
   {'eval_loss': 0.8612403869628906, 'eval_accuracy': 0.8342342342342343, 'eval_runtime': 244.7536, 'eval_samples_per_second': 9.07, 'eval_steps_per_second': 0.568, 'epoch': 8.0}                                                                                     
   {'train_runtime': 26052.4526, 'train_samples_per_second': 6.133, 'train_steps_per_second': 0.384, 'train_loss': 1.0575269655821498, 'epoch': 8.0}
   ***** eval metrics *****
   epoch                   =        8.0
   eval_accuracy           =     0.8342
   eval_loss               =     0.8612
   eval_runtime            = 0:04:41.22
   eval_samples_per_second =      7.894
   eval_steps_per_second   =      0.494
```
Language classification via common_language dataset (https://huggingface.co/datasets/common_language)

Jupyter notebook for finetune wave2vec2: signalAI/hfwave2vec2_finetune.ipynb

Pretrain the wave2vec2 model: signalAI/hfwave2vec2.py

Perform wave2vec2 training via "hfwave2vec2.py"
