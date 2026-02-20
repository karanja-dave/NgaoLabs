### import libraries for: 
#EDA
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

### Data preparation and cleaning 
# wrangle function
def wrangle(path):
    # load data 
    df=pd.read_csv(path)
    ## imputation of missing values 
    #categorical variables 
    cate_cols = ['travel_with', 'most_impressing'] #select cols with missing values 
    modes = df[cate_cols].mode().iloc[0] #get mode of these cols
    df[cate_cols] = df[cate_cols].fillna(modes) #impute missing values in these cols
    #numerical variables 
    cols = ['total_female', 'total_male']
    df[cols] = df[cols].fillna(df[cols].median())
    # drop high cardinality fetures 
    df.drop(columns='country',inplace=True)

    ## feature engineering 
    #total people
    # df["total_people"] = df["total_male"] + df["total_female"]
    #total nights
    # df["total_nights"] = df["night_mainland"] + df["night_zanzibar"]
    #drop cols used in feat engineering to avid multicolllinearity
    # df.drop(columns=['total_male','total_female','night_mainland','night_zanzibar'],inplace=True)

    return df 



# load data using wrangle func 
train=wrangle('data/Train.csv')
test=wrangle('data/Test.csv')


train.select_dtypes(include='object').nunique()
test.select_dtypes(include='object').nunique()

###EDA
##check data structure and format
train.head() #obv first 5 rows
train.info()

## check for missing values 
train.isna().mean()>0.5 
print('\n No variable has atleast 50% proportion of missing values.\n Imputation to be carried out on missing values\n')
train.isna().sum()



# examine cols with missing values individually 
#categorical examintaion
print('\nBefore imputation for categorical variables we need to know the mode which will be used for imputation\n')
train['travel_with'].unique() #check unique values in categorical variable
train['travel_with'].mode()[0] ##what is the most occurring unique value?

train['most_impressing'].unique()
train['most_impressing'].mode()[0]

#numerical examination
print("\nBefore imputation, we check for skewness in individual numerical variables.\nThis helps us know whetehr to use mean or median for imputation\n")
train['total_female'].skew()
train['total_male'].skew()
print('\nBoth numerical columns are right skewed. Imputation should be done using median\n')


# outliers 


## dealing with cardinality
train.select_dtypes(include='object').nunique()
train['country'].unique()
print('Country has high cardinality and should be dropped. \n But the again, it seems to be an important feature in the model')

# categorical analysis based on cardinality 
cate_var = train.select_dtypes(include="object")
for col in cate_var:
    print(f"{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts(dropna=False))
    print("-"*50)

print('\nfrom the analysis, I can observe that some categorical cols are ordered while other are not.\nThis implies that both label and one hot encoding will be applied to repsctive categorical variables\n')
train.info()

##Visualization
#dealing with multicollinearity
corr= train.select_dtypes("float64").corr()
sns.heatmap(corr)
plt.show()

print("No multicollinearity is observed betweeen any pair of feature variables")
# histogram : Distribution of numerical cols 
num_cols = train.select_dtypes(include="float64")
train[num_cols.columns].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()

num_cols.describe()

pd.read_csv('data/Train.csv')['total_cost'].head()

# data preprocessing 
# define categorical and label cols
categ_cols = train.select_dtypes(include='string').columns.drop('age_group')
label_cols = ["age_group"]

#init transformers
ohe = OneHotEncoder(drop='first', sparse_output=False)
le_feature = {col: LabelEncoder() for col in label_cols}

def feature_preprocessing(data, fit=True):
    data = data.drop(columns=['ID'], errors='ignore')
    
    #label encoding
    for col in label_cols:
        if fit:
            le_feature[col].fit(data[col].astype(str))
        data[col] = le_feature[col].transform(data[col].astype(str))
    
    # OHE
    if fit:
        ohe_array = ohe.fit_transform(data[categ_cols])
    else:
        ohe_array = ohe.transform(data[categ_cols])
    
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(categ_cols),
        index=data.index
    )
    
    # Drop original categorical columns and concatenate OHE
    data = data.drop(columns=categ_cols)
    data = pd.concat([data, ohe_df], axis=1)
    
    return data

feature_preprocessing(train, fit=True)
feature_preprocessing(test, fit=False)
