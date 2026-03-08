###Import libraries for 
# Data cleaning and EDA 
import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

# load data 
train = pd.read_csv('data/Train.csv').dropna()
test = pd.read_csv('data/Test.csv').dropna()


train.info()
train.head()
test.info()