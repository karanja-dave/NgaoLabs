###import libraries for
#EDA
import pandas as pd 
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



#data preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


## wrangle func 
#demographics 
def wrangle_demo(path):
    # load data 
    df=pd.read_csv(path)

    # drop features with high null count >0.5
    mask_na=df.isna().mean()>0.5    
    df.drop(columns=df.columns[mask_na],inplace=True)

    # impute missing values in employment status with mode employment status 
    mode_value = df['employment_status_clients'].mode()[0]
    df['employment_status_clients'] = df['employment_status_clients'].fillna(mode_value)

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

    # for now drop the time date cols 
    df.drop(columns=['approveddate','creationdate'],inplace=True)

    # relative risk
    # ompute average historical loan amount per customer
    df['avg_loan_amt'] = df.groupby('customerid')['loanamount'].transform('mean')
    # compute relative risk : loan to avergae loan ration/
    df['relative_risk'] = df['loanamount'] / df['avg_loan_amt']
    # drop `avg_loan_amt` col 
    df.drop(columns=['avg_loan_amt'], inplace=True)

    return df

##previous loans data 
def wrangle_prev_loans(path):
    # load data 
    df=pd.read_csv(path)
    #deal with high null counts
    mask_na=df.isna().mean()>0.5
    df.drop(columns=df.columns[mask_na],inplace=True)

    # select only date cols 
    mask_date=df.select_dtypes(include='object').drop(columns='customerid').columns
    # convert dates to date data type 
    for col in mask_date:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # loan approval period in hours: most loans are approved within a day
    df['approval_period'] = (df['approveddate'] - df['creationdate']).dt.total_seconds() / 3600

    # load default
    # loan repayment perido (days) : first loan
    df['first_delay'] = (df['firstrepaiddate'] - df['firstduedate']).dt.days 
    # subsequent loan delays (days)
    df['loan_delays'] = (df['closeddate'] - df['approveddate']).dt.days - df['termdays']
    # defaulted loans : Im using loans beyond dues date are defaulted 
    df['prev_loan_default'] = ((df['first_delay'] > 0) | (df['loan_delays'] > 0)).astype(int)
    #drop date cols, `first_delay` and `loan_delays`
    date_cols=df.select_dtypes(include='datetime').columns
    df.drop(columns=list(date_cols) + ['first_delay','loan_delays'],inplace=True)

    # relative risk
    # ompute average historical loan amount per customer
    df['avg_loan_amt'] = df.groupby('customerid')['loanamount'].transform('mean')
    # compute relative risk : loan to avergae loan ration/
    df['relative_risk'] = df['loanamount'] / df['avg_loan_amt']
    # drop `avg_loan_amt` col 
    df.drop(columns=['avg_loan_amt'], inplace=True)

    return df

### load data sets
##demographics data
demo_train=wrangle_demo('data/traindemographics.csv')
demo_test=wrangle_demo('data/testdemographics.csv')
# performance data 
perf_train=wrangle_perf('data/trainperf.csv')
perf_test=wrangle_perf('data/testperf.csv')

# previous loans data
prev_loans_train=wrangle_prev_loans("data/trainprevloans.csv") 
prev_loans_test=wrangle_prev_loans('data/testprevloans.csv')


### EDA 
##demographics 
demo_train.info()
demo_test.info()
demo_train.head()
demo_test.head()

# check for missing values
demo_train.isna().mean()>0.5
demo_train.isna().sum()
demo_test.isna().mean()>0.5
demo_test.isna().sum()

# dealing with cardinality
cate_var = demo_train.select_dtypes(include="object").drop(columns='customerid')
print("Cardinality count for categorical variables:\n")
cate_var.nunique()
print('\n `birthdate` will be used to get ages of bank clients. \n Transform it to date instead \n')
print("\n `customerid` is the primary key to be used to merge the 3 data-sets \n")
print("\n No high nor low cardinality is observed \n")
# detailed info on categorical variables
for col in cate_var:
    print(f"{col}: {demo_train[col].nunique()} unique values")
    print(demo_train[col].value_counts(dropna=False))
    print("-"*50)

##visulaization
# histogram : Distribution of numerical cols 
num_cols = demo_train.select_dtypes(include=["int64","float64"]).columns

demo_train[num_cols].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()

#age distribution with employment status
demo_train.boxplot(column='age', by='employment_status_clients', figsize=(12,6))
plt.title('Age Distribution by Employment Status')
plt.suptitle('')  
plt.ylabel('Age')
plt.xlabel('Employment Status')
plt.show()

# employment dist with bank type 
sns.countplot(data=demo_train, x='bank_account_type', hue='employment_status_clients')
plt.title('Employment Status Counts by Bank Account Type')
plt.xlabel('Bank Account Type')
plt.ylabel('Count')
plt.show()

# barchart
# func to plot barchart 
def plot_barh(df, col, ax):
    df[col].value_counts().plot(
        kind='bar',
        ax=ax,
        color='skyblue'
    )
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel("Count")
    ax.set_ylabel(col)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

plot_barh(demo_train, 'bank_account_type', axes[0])
plot_barh(demo_train, 'bank_name_clients', axes[1])
plot_barh(demo_train, 'employment_status_clients', axes[2])

plt.tight_layout()
plt.show()

# categorical encoding
# init ohe
ohe = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' avoids dummy variable trap for linear models

# select categorical columns for OHE
ohe_cols = ['bank_account_type', 'employment_status_clients', 'bank_name_clients']

# fit & transform train data
demo_train_ohe = pd.DataFrame(
    ohe.fit_transform(demo_train[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=demo_train.index
)

# transform test data using the same encoder
demo_test_ohe = pd.DataFrame(
    ohe.transform(demo_test[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=demo_test.index
)

# drop original categorical columns and concatenate OHE columns
demo_train = pd.concat([demo_train.drop(columns=ohe_cols), demo_train_ohe], axis=1)
demo_test = pd.concat([demo_test.drop(columns=ohe_cols), demo_test_ohe], axis=1)

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
perf_train.isna().sum()
perf_test.isna().sum()

# dealing with cardinality 
perf_train.select_dtypes(include='object')['good_bad_flag'].value_counts()
print("Only the target var is categorical and is of cardinality 2")

#visualization
#hist:dist of numerical variables
num_cols = perf_train.select_dtypes(include=["int64","float64"]).columns

perf_train[num_cols].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()

# barchart: dist for categorical var
(perf_train['good_bad_flag'].value_counts().
    plot( kind='bar',figsize=(6,4),color='skyblue', title=f'Distribution of Loan Defaulting'))

plt.xlabel("Defaulting")
plt.ylabel('Count') 
plt.show()

# Label encooding 
# split data 
X_train=perf_train.drop(['good_bad_flag'],axis=1)
y_train=perf_train['good_bad_flag']

## data preprocessing
# target encoding
le_target=LabelEncoder() 
y_train=le_target.fit_transform(y_train)
np.unique(y_train)


##previous loans data
prev_loans_train.info()
prev_loans_test.info()

# check for null counts 
prev_loans_train.isna().mean()>0.5
prev_loans_test.isna().mean()>0.5

# check if dates are of type date : no output as dates are dropped
prev_loans_train.select_dtypes(include='datetime').head()
prev_loans_test.select_dtypes(include='datetime').head()

#aggregate previous loan defaults per customer
customer_features = prev_loans_train.groupby('customerid').agg(
    num_bad_loans=('prev_loan_default', 'sum'),          # total previous defaults
    default_rate_prev=('prev_loan_default', 'mean'),    # fraction of loans defaulted
    recent_default_flag=('prev_loan_default', 'last')   # default status of most recent loan
).reset_index()

customer_features.describe()

## visualization
# hist :dist of all numerical vars 
num_cols = prev_loans_train.select_dtypes(include=["int64","float64"]).columns

prev_loans_train[num_cols].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()
