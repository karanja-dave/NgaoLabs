###import libraries for
#EDA
import pandas as pd 
from IPython.display import display

## wrangle func 
#demographics 
def wrangle_demo(path):
    # load data 
    df=pd.read_csv(path)

    # drop features with high null count >0.5
    mask_na=df.isna().mean()>0.5    
    df.drop(columns=df.columns[mask_na],inplace=True)

    # convert `datatype` to date
    df['birthdate'] = pd.to_datetime(df['birthdate'])
    # get everyone's age
    df['age']=(pd.Timestamp.today()-df['birthdate']).dt.days//365
    # drop `datatype` col 
    df.drop(columns='birthdate',inplace=True)

    return df

#performance data 
def wrangle_perf(path):
    # load data 
    df=pd.read_csv(path)
    # drop features with high null count
    mask_na=df.isna().mean()>0.5    
    df.drop(columns=df.columns[mask_na],inplace=True)
    # # convert `approvedate`&`creationdate` to date type
    # df['approveddate'] = pd.to_datetime(df['approveddate'])
    # df['creationdate'] = pd.to_datetime(df['creationdate'])


    return df

### load data sets
##demographics data
demo_train=wrangle_demo('data/traindemographics.csv')
demo_test=wrangle_demo('data/testdemographics.csv')
# performance data 
perf_train=wrangle_perf('data/trainperf.csv')
perf_test=wrangle_perf('data/testperf.csv')

# previous loans data
prev_loans_train=pd.read_csv("data/trainprevloans.csv") 
prev_loan_test=pd.read_csv('data/testprevloans.csv')


### EDA train 
##demographics 
demo_train.info()
demo_test.info()
demo_train.head()
demo_test.head()

# check for missing values
demo_train.isna().mean()>0.5
demo_test.isna().mean()>0.5

# dealing with cardinality
cate_var = demo_train.select_dtypes(include="object").drop(columns='customerid')
cate_var.nunique()
print('\n `birthdate` will be used to get ages of bank clients. \n Transform it to date instead \n')
print("\n `customerid` is the primary key to be used to merge the 3 data-sets \n")
print("\n No high nor low cardinality is observed \n")
# detailed info on categorical variables
for col in cate_var:
    print(f"{col}: {demo_train[col].nunique()} unique values")
    print(demo_train[col].value_counts(dropna=False))
    print("-"*50)

# outliers in age : age shld be b2n [18,80]
demo_train[~demo_train['age'].between(18, 80)]
demo_test[~demo_test['age'].between(18, 80)]
print("\n No unrealistic ages")

# outliers in location 
demo_train[
    ~(
        demo_train['longitude_gps'].between(2, 15) &
        demo_train['latitude_gps'].between(4.0, 14)
    )
]
print("\n In as much as we have some location oustside Nigeria's coordinates we can't drop them \n Rsn: They could be nigerians outside nigeria who took loans ")

##perfomamce data
perf_train.info()
perf_test.info()
print("\n `good_bad_flag` is the target variable \n don't drop bcz its not in the test!!!\n")

# check for missing values 
perf_train.isna().mean()>0.5
perf_test.isna().mean()>0.5

# dealing with cardinality 

perf_train.select_dtypes(include='object')['good_bad_flag'].value_counts()
perf_train