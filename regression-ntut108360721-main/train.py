#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
import xgboost as xgb
from hyperopt import hp, tpe, fmin
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

df_train = pd.read_csv('train-v3.csv')
df_test = pd.read_csv('test-v3.csv')
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
df_train.columns


# In[3]:


var = 'condition'
pri = 'price'
data = pd.concat([df_train[pri], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(5, 7))
fig = sns.boxplot(x=var, y=pri, data=data)
fig.axis(ymin=0, ymax=4000000);



# In[4]:


var = 'yr_built'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 8))
fig = sns.boxplot(x=var, y="price", data=data)
fig.axis(ymin=0, ymax=3000000);
plt.xticks(rotation=90);


# In[5]:


var = 'sqft_living'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(0,4000000),xlim=(0,10000));


# In[6]:


var = 'bedrooms'
data = pd.concat([df_train['sqft_living'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=var, y="sqft_living", data=data)
fig.axis(ymin=0, ymax=9000);
plt.xticks(rotation=90);


# In[7]:


var = 'bathrooms'
data = pd.concat([df_train['sqft_living'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=var, y="sqft_living", data=data)
fig.axis(ymin=0, ymax=11000);
plt.xticks(rotation=90);


# In[8]:


var = 'bathrooms'
data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=var, y="price", data=data)
fig.axis(ymin=0, ymax=6000000);
plt.xticks(rotation=90);


# In[9]:


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);


# In[10]:


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[11]:


sns.set()
cols = ['price', 'sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms', 'sqft_basement']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# In[12]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[13]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['price'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:20]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-20:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[14]:


var = 'sqft_living'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(0,8000000));


# In[15]:


var = 'grade'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(0,8000000));


# In[16]:


var = 'sqft_basement'

data = pd.concat([df_train['price'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(0,8000000));


# In[17]:


sns.distplot(df_train['price'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['price'], plot=plt)


# In[18]:


df_train['price'] = np.log(df_train['price'])
sns.distplot(df_train['price'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['price'], plot=plt)


# In[19]:


df_train['sqft_living'] = np.log(df_train['sqft_living'])
sns.distplot(df_train['sqft_living'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['sqft_living'], plot=plt)


# In[20]:



sns.distplot(df_train['sqft_basement'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['sqft_basement'], plot=plt)


# In[21]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['sqft_basement']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['sqft_basement']>0,'HasBsmt'] = 1


# In[22]:


df_train.loc[df_train['HasBsmt']==1,'sqft_basement'] = np.log(df_train['sqft_basement'])


# In[23]:


sns.distplot(df_train[df_train['sqft_basement']>0]['sqft_basement'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['sqft_basement']>0]['sqft_basement'], plot=plt)


# In[24]:


from sklearn.preprocessing import LabelEncoder

cols = ('sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms', 'sqft_basement')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[25]:


from scipy.stats import norm, skew #for some statistics
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[26]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# In[27]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[28]:


ntrain = ㄎtrain.shape[0]
ntest = test.shape[0]
y_train = train.price.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['price'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

train = all_data[:ntrain]
test = all_data[ntrain:]


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


y_train = train.price.values

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:




