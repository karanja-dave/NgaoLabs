### import libraries for: 
#EDA
import pandas as pd 
import pycountry 
import pycountry_convert as pc


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

    return df 



# load data using wrangle func 
train=wrangle('data/Train.csv')




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

## dealing with cardinality
train.select_dtypes(include='object').nunique()
train['country'].unique()
print('Country has high cardinality and should be dropped. \n But the again, it seems to be an important feature in the model')

# categorical analysis based on cardinality 
cate_var = train.select_dtypes(include="object").drop(columns=['ID','country'])
for col in cate_var:
    print(f"{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts(dropna=False))
    print("-"*50)

print('\nfrom the analysis, I can observe that some categorical cols are ordered while other are not.\nThis implies that both label and one hot encoding will be applied to repsctive categorical variables\n')
train.info()
 