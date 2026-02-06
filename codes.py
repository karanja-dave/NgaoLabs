## import libraries for:
# EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

###data preparation
# wrangle func used to load and clean data based on EDA findings 
def wrangle(filepath):
    # read csv files 
    df=pd.read_csv(filepath)
    # drop features with high/low cardinality 
    df.drop(columns=["uniqueid"],inplace=True)

    return df
# load data 
train=wrangle("data/Train.csv")
test=wrangle("data/Test.csv")
print(train.head())
print(test.head())

## Exploratory Data Analysis
# checl dataset shape 
print(f"Train dataset:{train.shape[0]} rows, {train.shape[1]} columns")
print(f"Test dataset:{test.shape[0]} rows, {test.shape[1]} columns")
# check for missing values 
print('missing values:', train.isnull().sum())
print("There are no missing values in all the variables (feature+target)")
# check structure of data 
train.info()
print("From the data structure, we have, 0nly 3 variables as integers, the rest are strings (probably categorical variables)")
# Dealing with high and low cardinality 
train.select_dtypes(include="object").nunique()
print("We'll drop `uniqueid` as it has high cardinality")
# delaing with multi-collinearity
corr= train.select_dtypes("number").corr()
sns.heatmap(corr)
plt.show()
print("No multicollinearity is observed betweeen any pair of feature variables")

#Visulaization
sns.catplot(x="bank_account",kind="count",data=train)


## data preprocessing
# target encoding
le_target=LabelEncoder() 
train["bank_account"]=le_target.fit_transform(train["bank_account"])
train["bank_account"].unique()

# feature encoding 
categ_cols = ["relationship_with_head","marital_status",
    "education_level","job_type","country"]

label_cols = ["location_type","cellphone_access","gender_of_respondent"]

num_cols = ["household_size", "age_of_respondent", "year"]

# preprocessing func for features 
def feature_preprocessing(data):
    # one hot encoding 
    data = pd.get_dummies(data, prefix_sep="_",columns=categ_cols)
    # label encoding
    for col in label_cols:
        data[col]=le_target.fit_transform(data[col])
    # min max scaling 
    # scaler=MinMaxScaler(feature_range=(0,1))
    # data=scaler.fit_transform(data)

    return data 

feature_preprocessing(train)
    
###model building
##model evaluation, performance and selection

###Communicating results
