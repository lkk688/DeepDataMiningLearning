#ref: https://www.analyticsvidhya.com/blog/2023/06/building-a-multi-task-model-for-fake-and-hate-probability-prediction-with-bert/

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

if __name__ == "__main__":
    df = pd.read_csv('./sampledata/fake-hate.csv') 
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle the dataset
    print(df.head())