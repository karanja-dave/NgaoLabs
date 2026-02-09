## import libraries for:
# EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier



# performance metrics 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# data preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# feature selection 
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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

# split data 
X_train=train.drop(['bank_account'],axis=1)
y_train=train['bank_account']

## data preprocessing
# target encoding
le_target=LabelEncoder() 
y_train=le_target.fit_transform(y_train)
# feature encoding 
categ_cols = ["relationship_with_head","marital_status",
    "education_level","job_type","country"]

label_cols = ["location_type","cellphone_access","gender_of_respondent"]

num_cols = ["household_size", "age_of_respondent", "year"]

# preprocessing func for features _
# initialize transformers 
scaler=MinMaxScaler()
le_feature = {col: LabelEncoder() for col in label_cols}


def feature_preprocessing(data, fit=True):
    # One-hot encoding
    data = pd.get_dummies(data, prefix_sep="_", columns=categ_cols)
    
    # Label encode label_cols
    for col in label_cols:
        if fit:
            le_feature[col].fit(data[col])
        data[col] = le_feature[col].transform(data[col])
    
    # MinMax scaling
    if fit:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    return pd.DataFrame(data_scaled,columns=data.columns)


# pre-process train and test data 
processed_train = feature_preprocessing(X_train, fit=True)
processed_test = feature_preprocessing(test, fit=False)

#split train data 
X_Train1, X_val, y_Train, y_val= train_test_split(processed_train,y_train,stratify=y_train,
                                                 test_size=0.1,random_state=42)
## feature selection
#filter method :to remove obvious irrelevant features 
skb = SelectKBest(score_func=mutual_info_classif, k=20)
skb.fit(X_Train1, y_Train)
scores = pd.DataFrame(skb.scores_)
cols = pd.DataFrame(X_Train1.columns)
featureScores = pd.concat([cols, scores], axis=1)
featureScores.columns = ['feature', 'score']
featureScores.nlargest(37, 'score')
selected_features = featureScores[featureScores['score'] > 0.001]

#drop irreleveant featres
kept_features= selected_features['feature'].tolist()

X_Train=X_Train1[kept_features]
X_val = X_val[kept_features]
X_test=processed_test[kept_features]

    
###model building

##logistics regression
#baseline model
baseline_lr=LogisticRegression(max_iter=1000,random_state=123)
baseline_lr.fit(X_Train,y_Train)
y_pred_base=baseline_lr.predict(X_val)
y_prob_base=baseline_lr.predict_proba(X_val)[:,1]

#baseline evaluation
acc = accuracy_score(y_val, y_pred_base)
f1 = f1_score(y_val, y_pred_base)
auc = roc_auc_score(y_val, y_prob_base)

acc, f1, auc

#iteration 1
# remove features with zero coeeficients using lasso 1r
lr_lasso= LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=123)
lr_lasso.fit(X_Train,y_Train)
y_pred_l1 = lr_lasso.predict(X_val)
y_prob_l1 = lr_lasso.predict_proba(X_val)[:, 1]

# evaluate the lasso lr 
acc_l1 = accuracy_score(y_val, y_pred_l1)
f1_l1 = f1_score(y_val, y_pred_l1)
auc_l1 = roc_auc_score(y_val, y_prob_l1)

acc_l1, f1_l1, auc_l1
print("lasso lr slightly improves F1 and accuracy,maintains similar ROC-AUC, showing better performance over the baseline.")
print("The model does not overfit nor underfit as: training (baseline) and validation(lasso LR) are close")

# select features with significant coeff
coef = pd.Series(lr_lasso.coef_[0], index=X_Train.columns)
l1_feat = coef[coef != 0].index.tolist()
#features slected from lasso lr 
X_TrainL1 = X_Train[l1_feat]
X_valL1 = X_val[l1_feat]
print("A total of 5 features eliminated by lasso lr. fit one more LR without them, see if performance imporves")

# iteration 2
lr_l1_reduced = LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000,random_state=123)
lr_l1_reduced.fit(X_TrainL1,y_Train)
y_pred_l1_reduced = lr_l1_reduced.predict(X_valL1)
y_prob_l1_reduced = lr_l1_reduced.predict_proba(X_valL1)[:, 1]

# evaluate the lasso lr 
acc_l1_reduced = accuracy_score(y_val, y_pred_l1_reduced)
f1_l1_reduced = f1_score(y_val, y_pred_l1_reduced)
auc_l1_reduced = roc_auc_score(y_val, y_prob_l1_reduced)

acc_l1_reduced, f1_l1_reduced, auc_l1_reduced
print("No much difference from the prior models!!")

##Random Forest
#baseline model
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=123) #init model
# fit baseline
rf_baseline.fit(X_Train,y_Train)
# prediction
y_pred_rf=rf_baseline.predict(X_val)
y_prob_rf=rf_baseline.predict_proba(X_val)[:, 1]
# evalute
acc_rf=accuracy_score(y_val,y_pred_rf)
f1_rf=f1_score(y_val, y_pred_rf)
auc_rf=roc_auc_score(y_val, y_prob_rf)

acc_rf,f1_rf,auc_rf

# select significant features 
signif_feat=pd.Series(rf_baseline.feature_importances_,index=X_Train.columns).sort_values(ascending=False)
# remove feat with <0.01 significance to target contribution
rf_feat = signif_feat[signif_feat > 0.01].index.tolist()
# Reduce train and validation sets
X_TrainRF = X_Train[rf_feat]
X_valRF = X_val[rf_feat]

# iteration 1
rf_reduced1 = RandomForestClassifier(n_estimators=300,random_state=123,n_jobs=-1)
#fit
rf_reduced1.fit(X_TrainRF, y_Train)
#predict
y_pred_rf1 = rf_reduced1.predict(X_valRF)
y_prob_rf1 = rf_reduced1.predict_proba(X_valRF)[:, 1]
# evaluate
acc_rf1 = accuracy_score(y_val, y_pred_rf1)
f1_rf1  = f1_score(y_val, y_pred_rf1)
auc_rf1 = roc_auc_score(y_val, y_prob_rf1)

acc_rf1, f1_rf1, auc_rf1
print("The reduced Random Forest model outdoes the baseline by a small margin")

##Gradient boosting
# baseline
# init model 
gb_baseline = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=123)
# fit model 
gb_baseline.fit(X_Train, y_Train)
# predictions
y_pred_gb = gb_baseline.predict(X_val)
y_prob_gb = gb_baseline.predict_proba(X_val)[:, 1]
# evaluate
acc_gb = accuracy_score(y_val, y_pred_gb)
f1_gb = f1_score(y_val, y_pred_gb)
auc_gb = roc_auc_score(y_val, y_prob_gb)

acc_gb, f1_gb, auc_gb
# feature selection 
signif_feat_gb=pd.Series(gb_baseline.feature_importances_, index=X_Train.columns).sort_values(ascending=False)
signif_feat_gb = signif_feat_gb[signif_feat_gb > 0.01].index.tolist()

# feature reduction 
X_TrainGB = X_Train[signif_feat_gb]
X_valGB = X_val[signif_feat_gb]

# iteration 1
gb_reduced=GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=123)
# fit
gb_reduced.fit(X_TrainGB, y_Train)
#predict
y_pred_gb1 =gb_reduced.predict(X_valGB)
y_prob_gb1 =gb_reduced.predict_proba(X_valGB)[:, 1]
# evaluate
acc_gb1 = accuracy_score(y_val, y_pred_gb1)
f1_gb1  = f1_score(y_val, y_pred_gb1)
auc_gb1 = roc_auc_score(y_val, y_prob_gb1)

acc_gb1, f1_gb1, auc_gb1

##k nearest neighbors
#init  baseline
knn_baseline = KNeighborsClassifier(n_neighbors=5) 
knn_baseline.fit(X_Train, y_Train)

# 2. Predictions and evaluation
y_pred_knn = knn_baseline.predict(X_val)
y_prob_knn = knn_baseline.predict_proba(X_val)[:, 1]

acc_knn = accuracy_score(y_val, y_pred_knn)
f1_knn = f1_score(y_val, y_pred_knn)
auc_knn = roc_auc_score(y_val, y_prob_knn)

acc_knn, f1_knn, auc_knn

# 3. Iteration: test different k-values
k_values = [3, 5, 7, 9, 11, 13]
results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_Train, y_Train)
    y_pred = knn.predict(X_val)
    y_prob = knn.predict_proba(X_val)[:, 1]
    results[k] = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_prob)
    }

results

###Communicating results
