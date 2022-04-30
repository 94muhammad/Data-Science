#!/usr/bin/env python
# coding: utf-8

# # Regression

# In[1]:


# import libraries

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import skew, norm
from sklearn.model_selection import train_test_split

import lightgbm as lgb


# In[2]:


import warnings

def ignore_warnings(*wargs, **kwargs):
    pass

warnings.warn = ignore_warnings


# In[3]:


# load csv file 
df = pd.read_csv('train.csv')


# In[4]:


df.head()


# In[5]:


# dataframe desciption (count, mean, etc)
df.describe()


# In[6]:


# check column's data-type and null value presence
df.info()


# In[7]:


# total null values in columns
df.isnull().sum()


# In[8]:


# plot columns with % of null values
def plot_nas(df):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' : na_df})
        missing_data.plot(kind = "barh")
        plt.show()
    else:
        print('No NAs found')
plot_nas(df)


# In[9]:


df.drop('Id', axis = 1, inplace=True)
print("Shape of teh dataframe is : {}".format(df.shape))


# In[10]:


# fill null with 0 for numerical columns
for i in df.columns:
    if df[i].dtype == 'int64' or df[i].dtype == 'float64':
        df[i] = df[i].fillna(0)


# In[11]:


# fill object columns having null values less than 5 with mode()
for col in df.columns:
    if df[col].isnull().sum() <= 5:
        df[col] = df[col].fillna(df[col].mode()[0])


# In[12]:


# for example in "Electrical" column, SBrkr is most commonly used and had one null value which is replaced with SBrkr
df['Electrical']


# In[13]:


# for object columns with higher number of null values, 'None' is inserted
for col in df.columns:
    if df[col].dtype == 'O':
        if df[col].isnull().sum() > 5:
            df[col] = df[col].fillna('None')


# In[14]:


# To check if there is any missing data

df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
missing_data.head()


# In[15]:


#visualizing square footage of (home,lot,above and basement)
fig = plt.figure(figsize=(16,5))
fig.add_subplot(2,2,1)
sns.scatterplot(df['1stFlrSF'], df['SalePrice'])
fig.add_subplot(2,2,2)
sns.scatterplot(df['GrLivArea'],df['SalePrice'])
fig.add_subplot(2,2,3)
sns.scatterplot(df['TotalBsmtSF'],df['SalePrice'])
fig.add_subplot(2,2,4)
sns.scatterplot(df['GarageArea'],df['SalePrice'])


# In[16]:


# Outlier removal from target
def scatter_plot(x,y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show


# In[17]:


scatter_plot(df['GrLivArea'], df['SalePrice'])


# In[18]:


# we can see outlier presence from the plot above which indicates the presence of low sale price for large area
# which don't conform to the rest of data pattern. So we will remove it
df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

scatter_plot(df['GrLivArea'], df['SalePrice'])


# In[19]:


# Correlation
df.corr()


# In[20]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), square=True)


# In[21]:


df.corrwith(df["SalePrice"])


# In[22]:


plt.figure(figsize=(8, 12))

heatmap = sns.heatmap(df.corr()[["SalePrice"]].sort_values(by='SalePrice', ascending=False))
heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16)


# In[23]:


sns.distplot(df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot to observe the distribution
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# In[24]:


# use the numpy fuction log1p which  applies log(1+x) to all elements of the column
df["SalePrice"] = np.log1p(df["SalePrice"])

#Check the new distribution 
sns.distplot(df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.show()


# In[25]:


df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']


# In[26]:


# transform numerical variable which represents category
col = ['MSSubClass','OverallCond','YrSold','MoSold']
for i in col:
    df[i]=df[i].apply(str)


# In[27]:


# Label Encode categorical vriables
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'O':
        encoder.fit(list(df[col].values))
        df[col] = encoder.transform(list(df[col].values))

print("Transformed data shape {}".format(df.shape))


# In[28]:


# #check for data skewness

# numeric_data = df.dtypes[df.dtypes != "object"].index

# # Check the skew of all numerical features
# skewed_data = df[numeric_data].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_data})
# skewness


# In[29]:


# plot correlation after feature engineering to check the relation of features with target
plt.figure(figsize=(8, 12))

heatmap = sns.heatmap(df.corr()[["SalePrice"]].sort_values(by='SalePrice', ascending=False))
heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16)


# In[30]:


df_train_test = df.drop(['SalePrice'], axis=1)
df_target = df['SalePrice']


# In[31]:


df_target


# In[32]:


df_train_test


# ### LightGBM Regressor

# In[33]:


X_train, x_test, Y_train, y_test = train_test_split(df_train_test, df_target, test_size=0.2, random_state=0)


# In[34]:


# # one hot encoding using keras
# from numpy import array
# from numpy import argmax
# from keras.utils import to_categorical

# encode_ohe = to_categorical(df_train)


# In[35]:


# print(encode_ohe.shape)


# In[36]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# x_test = scaler.transform(x_test)


# In[38]:


Y_train


# In[39]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=920,
                              max_bin = 55,
                            feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=10,
                              min_data_in_leaf =5)

model_lgb.fit(X_train, Y_train)


# In[40]:


lgb_preds = model_lgb.predict(x_test)


# In[41]:


from sklearn import metrics 
print(metrics.r2_score(y_test, lgb_preds))


# In[42]:


print(metrics.mean_squared_error(y_test,lgb_preds))


# In[ ]:




