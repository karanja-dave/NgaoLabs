### import libraries for: 
#EDA
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#preprocessing 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold,  train_test_split, GridSearchCV
#model building 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from xgboost import XGBRegressor





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
    # df.drop(columns='country',inplace=True)

    ## feature engineering 
    #total people
    df["total_people"] = df["total_male"] + df["total_female"]
    #total nights
    df["total_nights"] = df["night_mainland"] + df["night_zanzibar"]

    # count total packages 
    pkg_cols = df.filter(like='package_').columns #select cols with package name 
    df['package_count'] = (df[pkg_cols] == 'Yes').sum(axis=1)

    # ratios
    df['mainland_ratio'] = df['night_mainland'] / (df['total_nights'] + 1e-6)
    df['zanzibar_ratio'] = df['night_zanzibar'] / (df['total_nights'] + 1e-6)
    
    #drop cols used in feat engineering to avid multicolllinearity
    df.drop(columns=list(pkg_cols)+['total_male', 'total_female', 'night_mainland', 'night_zanzibar'],inplace=True)
    #drop one ratio to avoid multicollinearity as they have very high correlation
    df.drop(columns='zanzibar_ratio',inplace=True)

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
print('\nCountry has high cardinality and should be dropped. \n But the again, it seems to be an important feature in the model')
print('\nWe will not be droppping country but we will be transforming it into numerical values using `Cross-Validated Target Encoding` explained well in the documentation')

# categorical analysis based on cardinality 
cate_var = train.select_dtypes(include="object")
for col in cate_var:
    print(f"{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts(dropna=False))
    print("-"*50)

print('\nfrom the analysis, I can observe that some categorical cols are ordered while other are not.\nThis implies that both label and one hot encoding will be applied to repsctive categorical variables\n')
train.info()

# dealing with incosistencies in the `age_group` col 
test['age_group'] = test['age_group'].replace({'24-Jan': '1-24'})

##Visualization
#dealing with multicollinearity
corr= train.select_dtypes("float64").corr()
sns.heatmap(corr)
plt.show()

print("\n No multicollinearity is observed betweeen any pair of feature variables\n However theres a high correlation between the engineered ratios, so dropped one of them\n")
# histogram : Distribution of numerical cols 
num_cols = train.select_dtypes(include="float64")
train[num_cols.columns].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()

num_cols.describe()

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

fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # 3 rows × 3 columns
axes = axes.flatten()

plot_barh(train, 'age_group', axes[0])
plot_barh(train, 'travel_with', axes[1])
plot_barh(train, 'purpose', axes[2])
plot_barh(train, 'main_activity', axes[3])
plot_barh(train, 'info_source', axes[4])
plot_barh(train, 'tour_arrangement', axes[5])
plot_barh(train, 'payment_mode', axes[6])
plot_barh(train, 'first_trip_tz', axes[7])
plot_barh(train, 'most_impressing', axes[8])

plt.tight_layout()
plt.show()

#boxplot
# select numerical columns
num_cols = train.select_dtypes(include='float64').columns

# create subplots grid
n_cols = 3  # number of columns in the grid
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # calculate required rows
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()  # flatten for easy indexing

# plot boxplots for each numerical column
for i, col in enumerate(num_cols):
    train.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

# remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#scatter plot 
# List of numerical features to plot against total_cost
num_vars = ['total_people', 'total_nights', 'mainland_ratio']

# Create a grid of scatter plots
fig, axes = plt.subplots(1, len(num_vars), figsize=(18, 5))

for i, var in enumerate(num_vars):
    sns.scatterplot(data=train, x=var, y='total_cost', ax=axes[i])
    axes[i].set_title(f'{var} vs Total Cost')

plt.tight_layout()
plt.show()

### data preprocessing 
##encoding the country column
#train
train['country_encoded'] = np.nan
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(train):
    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]
    
    country_mean = train_fold.groupby('country')['total_cost'].mean()
    
    train.loc[val_idx, 'country_encoded'] = val_fold['country'].map(country_mean)

# fill missing/unseen categories with global mean
train['country_encoded']=train['country_encoded'].fillna(train['total_cost'].mean())
# drop `country` col
train.drop(columns='country',inplace=True)

#test encoding of country column
test['country_encoded'] = test['country'].map(country_mean)
#fill missing values 
test['country_encoded']=test['country_encoded'].fillna(train['total_cost'].mean())
# drop `country`col 
test.drop(columns='country',inplace=True)



# define categorical and label cols
categ_cols = train.select_dtypes(include='string').columns.drop(['age_group', 'ID'])
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

#separate target and feature variables 
X_train=train.drop(['total_cost'],axis=1)
y_train=train['total_cost']

#encoding trnasformation
encoded_train=feature_preprocessing(X_train, fit=True)
encoded_test=feature_preprocessing(test, fit=False)


# split train and validation sets 
X_Train1, X_val, y_Train, y_val = train_test_split(
    encoded_train, y_train, test_size=0.1, random_state=42
)

### Model Building 
## linear regression baseline
lr_base = LinearRegression() #init model
# fit
lr_base.fit(X_Train1, y_Train)
# predict
y_pred = lr_base.predict(X_val)
# evalutaion 
rmse_lr_base = np.sqrt(mean_squared_error(y_val, y_pred))
mae_lr_base = mean_absolute_error(y_val, y_pred)
r2_lr_base = r2_score(y_val, y_pred)

print("Baseline Linear Regression")
rmse_lr_base,mae_lr_base,r2_lr_base

# residual diagonistics 
residuals = y_val - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
print('\nScatter of residual vs prediction is funnel shaped.\n This means the  variance increases as predicted values increase.\n This strongly supports log-transforming the target\n.')
print('\nResults suggest log transformation of target variable')

# residual dist 
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.show()
print('\nDistribution of residuals is right skewed confirming non normality.\nThis seconds log transformation of target variable')

# inspect coeeficients 
pd.DataFrame({
    "Feature": X_Train1.columns,
    "Coefficient": lr_base.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

## iteration 1: log transformation of target variable
# log-transformation of target 
y_train_log = np.log1p(y_Train)
y_val_log = np.log1p(y_val)
# init model 
lin_log = LinearRegression()
# fit model 
lin_log.fit(X_Train1, y_train_log)
# prediction
y_pred_log = lin_log.predict(X_val)
# reconvert target predictions from log 
y_pred = np.expm1(y_pred_log)
# evaluation
rmse_lr_log = np.sqrt(mean_squared_error(y_val, y_pred))
mae_lr_log = mean_absolute_error(y_val, y_pred)
r2_lr_log = r2_score(y_val, y_pred)

rmse_lr_log,mae_lr_log,r2_lr_log

##decison trees 
# baseline
dt_base = DecisionTreeRegressor(random_state=42)
dt_base.fit(X_Train1, y_Train)

# Predictions
y_pred = dt_base.predict(X_val)

# Performance metrics
rmse_dt_base = np.square(mean_squared_error(y_val, y_pred))
mae_dt_base = mean_absolute_error(y_val, y_pred)
r2_dt_base = r2_score(y_val, y_pred)

rmse_dt_base,mae_dt_base,r2_dt_base
print('Baseline performed worser that the regression models, lets tune its hyper-paremters')

# iteration 
# define parameter grid 
param_grid = {'max_depth': [3, 5, 7, 10, None],'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']}

# define RMSE scorer (negate because GridSearchCV maximizes the score)
rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
# grid Search
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,scoring=rmse_scorer,
    cv=5,n_jobs=-1)

# Fit GridSearch to training data
grid_search.fit(X_Train1, y_Train)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# Evaluate on test set
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_val)

rmse_dt = np.square(mean_squared_error(y_val, y_pred))
mae_dt = mean_absolute_error(y_val, y_pred)
r2_dt = r2_score(y_val, y_pred)

rmse_dt,mae_dt,r2_dt


##XGBOOST
# init model 
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=500, 
    learning_rate=0.05, max_depth=5, subsample=0.8,
    colsample_bytree=0.8,random_state=42)

# Fit on training data
xgb_model.fit(X_Train1, y_Train)

# Predict on validation set
y_pred_xgb = xgb_model.predict(X_val)

# Evaluate performance
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)

rmse_xgb, mae_xgb, r2_xgb

# Predict total_cost on test set
test['total_cost'] = xgb_model.predict(encoded_test)

# Create submission DataFrame with only ID and predicted total_cost
submission = pd.DataFrame({
    "ID": test["ID"],
    "total_cost": test['total_cost']
})

# Preview 5 samples
submission.sample(5)

# Export to CSV
submission.to_csv('first_submission.csv', index=False)