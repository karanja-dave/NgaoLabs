###Import libraries for 
# 1.EDA 
import pandas as pd 
import numpy as np 
# 2.visulaization 
import matplotlib.pyplot as plt 
import seaborn as sns
# 3. preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# 4. evaluation 
from sklearn.metrics import silhouette_score

# setting pandas to show all rows and columns 
# Show all rows
pd.set_option('display.max_rows', None)
# Show all columns
pd.set_option('display.max_columns', None)
# Optional: widen column display
pd.set_option('display.width', 200)








### Data cleaning and wrangling 
def wrangle(path):
    # load data 
    df=pd.read_csv(path)

    # convert date cols to date type 
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # drop leakages 
    df=df.drop(columns=['target_min','target_max','target_variance','target_count'],errors='ignore')

    # drop cols with high null count 
    mask_na=df.isna().mean()>0.5
    df.drop(columns=df.columns[mask_na],inplace=True)

    ## imputation
    # select cols with low null count for imputation 
    num_cols_na = df.columns[df.isna().sum() > 0]
    # compute skewness-helps decided cols to be imputed with mean or medain
    skew = df[num_cols_na].skew()
    # separate cols : those to be imputed with mean from those imputed with median
    mean_cols = skew[skew.abs() < 0.5].index
    median_cols = skew[skew.abs() >= 0.5].index
    # compute stats from df
    mean_values = df[mean_cols].mean()
    median_values = df[median_cols].median()
    # imputation with respective stats 
    df[mean_cols] = df[mean_cols].fillna(mean_values)
    df[median_cols] = df[median_cols].fillna(median_values)
    ## dealing with multicollinearity
    #multicollinearity on geometry cols (angle columns)
    sat_cols=df.filter(regex='(O3|CO|HCHO|SO2|CLOUD|AER_AI)_(sensor|solar)_(azimuth|zenith)_angle').columns
    df=df.drop(columns=df[sat_cols])



    return df

# load train and tes 
train=wrangle('data/Train.csv')
test=wrangle('data/Test.csv')


###EDA
#checks top 5 rows 
train.head()
train.info()
test.info()

# check dim - ensures that test and train have same number of variables
train.shape
test.shape

print('\n Train has more variables than test, thus drop variables in train not in test\n')

# check for cols in train and not in test 
set(train.columns) - set(test.columns)
print("\n most of this columns were identified as leakages to the target and were dropped\n Don't drop the the target variable\n")

# check for high null counts 
train.isna().mean()>0.5
print("\n Now that we've dropped columns with high null counts,\n let's check for cols with low null counts and,\n impute them with median or mean based on their distributions")

#check cols with missing values (low null counts that can be imputed)
train.isna().sum()

# filter out cols with missing values 
num_cols_na = train.columns[train.isna().sum() > 0]
# hist plot to show distribution 
train[num_cols_na].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()
print("We have so many variables that Visulization is not giving, lets use descriptive stats to analyze numerical cols")

# function for descriptive stats on numerical cols 
def desc_stats(df):
    # select only numerical cols 
    num_df = df.select_dtypes(include=np.number)
    # descriptive stats calc 
    summary = pd.DataFrame({
        "count": num_df.count(),
        "missing": num_df.isna().sum(),
        "missing_pct": num_df.isna().mean(),
        "mean": num_df.mean(),
        "median": num_df.median(),
        "std": num_df.std(),
        "min": num_df.min(),
        "q1": num_df.quantile(0.25),
        "q3": num_df.quantile(0.75),
        "max": num_df.max(),
        "skew": num_df.skew()
    })
    
    return summary.sort_values("missing_pct", ascending=False)
# descriptive stats on whole train set  
desc_stats(train)
desc_stats(test)
# descriptive stats on numerical cols with missing values 
desc_stats(train[num_cols_na])

###dealing with multicollinearity 
#select all numerical cols
num_cols = train.select_dtypes(include=["float64"]).columns

## get correlation of satellite columns
gas_cols = train[num_cols].filter(regex='_(NO2|O3|CO|HCHO|SO2)_(sensor|solar)_(zenith|azimuth)_angle').columns
train[gas_cols].corr()
print("Highly correlated geometry variables across gases were removed. Only NO2 sensor and solar angles were kept since satellite viewing geometry is identical across gas products.")
#get correlation of cloud + NO2+AER_AI (aerosol index)
sat_cols=train[num_cols].filter(regex='_(NO2|CLOUD|AER_AI)_(sensor|solar)_(azimuth|zenith)_angle').columns
train[sat_cols].corr()
print("We'll Drop all 8 redundant angle columns (r>0.9), only keeping 4 NO2 angles for PM2.5 relevance")
# gte correlation of NO2 
sat_cols=train[num_cols].filter(regex='NO2_(sensor|solar)_(azimuth|zenith)_angle').columns
corr=train[sat_cols].corr()
sns.heatmap(corr)
plt.show()

###Visualization
# distribution of numerical cols 

train[num_cols].hist(bins=30, figsize=(14,10))
plt.tight_layout()
plt.show()

###Unsupervised Learning, Clustering
# separate features from target, drop Id and Date 
X_train=train.drop(columns=['target','Place_ID X Date','Date','Place_ID'])
y_train=train['target']

#scaling-clustering depends on distrance 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


##k-means clustering (elbow method)
inertia_vals=[]
sil_scores=[]

for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    inertia_vals.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Elbow plot
axes[0].plot(range(2,11), inertia_vals, marker='o')
axes[0].set_xlabel("Number of clusters")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

# Silhouette plot
axes[1].plot(range(2,11), sil_scores, marker='o')
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")

plt.tight_layout()
plt.show()

print('\n Optimal number pf clusters is where k=5\n This is supported by the elbow method and the silhouette analysis\n')

# optimal k-clustering 
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit_predict(X_scaled)
train['cluster']=kmeans.labels_
train['cluster'].value_counts().sort_index()

##hierarchical clustering
hc = linkage(X_scaled, method='ward')


# plot the Dendrogram
plt.figure(figsize=(12, 7))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index (or Place_ID)")
plt.ylabel("Distance (Ward)")

dendrogram(hc,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,                   # show 20 clusters for readability
    show_leaf_counts=True,leaf_rotation=90.,
    leaf_font_size=12.,show_contracted=True,)

#draw the "Cut Line"
# Adjust this height to see how many clusters it creates
plt.axhline(y=15, color='r', linestyle='--') 
plt.show()

X_scaled.to_csv('sample.csv',index=False)


for train_idx, val_idx in folds.split(train, pd.qcut(target, 10, labels=False)):

    X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators=2200)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=200,
        verbose=False
    )

    oof[val_idx] = model.predict(X_val)
    test_pred += model.predict(test[features]) / folds.n_splits