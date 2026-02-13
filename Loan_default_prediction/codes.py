###import libraries for
#EDA
import pandas as pd 
from IPython.display import display



### load data sets
##train set 
#demographic data
demo_train=pd.read_csv('data/traindemographics.csv')
# performance data 
perf_train=pd.read_csv('data/trainperf.csv')
# previous loans data
prev_loans_train=pd.read_csv("data/trainprevloans.csv")

##test set
# demographics
demo_test=pd.read_csv('data/testdemographics.csv')
# performance data 
perf_test=pd.read_csv('data/testperf.csv')
# previous loans data 
prev_loan_test=pd.read_csv('data/testprevloans.csv')


### EDA train 
##demographics 
demo_train.info()
demo_test.info()

# check for missing values
demo_train.isna().sum()
demo_test.isna().sum()

# dealing with cardinality
categorical_cols = demo_train.select_dtypes(include="object")
categorical_cols.nunique()
print('\n birthdate will be used to get ages of bank clients. \n Transform it to date instead')

# detailed info on categorical variables
for col in categorical_cols:
    print(f"{col}: {demo_train[col].nunique()} unique values")
    print(demo_train[col].value_counts(dropna=False))
    print("-"*50)