#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[2]:


data = pd.read_csv('adult.csv')
data.head()


# In[3]:


data.columns


# In[4]:


data = data.rename(columns={'marital-status':'marital_status'})
data = data.rename(columns={'native-country':'native_country'})
data


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.education.value_counts()


# In[8]:


data.occupation.value_counts()


# In[9]:


data.income.value_counts()


# In[10]:


pd.get_dummies(data.occupation)


# In[11]:


data = pd.concat([data.drop('occupation',axis=1),pd.get_dummies(data.occupation).add_prefix('occupation_')],axis=1)
data = pd.concat([data.drop('workclass',axis=1),pd.get_dummies(data.workclass).add_prefix('workclass_')],axis=1)
data = data.drop('education',axis=1)
data = pd.concat([data.drop('marital_status',axis=1),pd.get_dummies(data.marital_status).add_prefix('marital_status_')],axis=1)
data = pd.concat([data.drop('relationship',axis=1),pd.get_dummies(data.relationship).add_prefix('relationship_')],axis=1)
data = pd.concat([data.drop('native_country',axis=1),pd.get_dummies(data.native_country).add_prefix('native_country_')],axis=1)
data = pd.concat([data.drop('race',axis=1),pd.get_dummies(data.race).add_prefix('race_')],axis=1)
data


# In[12]:


data['gender']= data['gender'].apply(lambda x: 1 if x=='Male' else 0)
data['income']= data['income'].apply(lambda x: 1 if x=='>50K' else 0)
data


# In[13]:


plt.figure(figsize=(18,12))
sns.heatmap(data.corr(), annot=False, cmap='YlGnBu')


# In[14]:


data.corr()


# In[15]:


correlations = data.corr()['income'].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8 * len(data.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
data_dropped = data.drop(cols_to_drop, axis = 1)


# In[16]:


plt.figure(figsize=(20,16))
sns.heatmap(data_dropped.corr(), annot=True, cmap='YlGnBu')


# In[17]:


data=data.drop('fnlwgt', axis=1)


# In[18]:


train_data, test_data = train_test_split(data, test_size=0.2)


# In[19]:


train_data


# In[20]:


test_data


# In[21]:


train_X = train_data.drop('income',axis=1)
train_y = train_data['income']

test_X=test_data.drop('income', axis=1)
test_y=test_data['income']


# In[22]:


RandomForest= RandomForestClassifier()
RandomForest.fit(train_X, train_y)


# In[23]:


RandomForest.score(test_X,test_y)


# In[24]:


RandomForest.feature_importances_


# In[25]:


RandomForest.feature_names_in_


# In[26]:


dict(zip(RandomForest.feature_names_in_, RandomForest.feature_importances_))


# In[27]:


importances = dict(zip(RandomForest.feature_names_in_, RandomForest.feature_importances_))
importances= {k: v for k, v in sorted(importances.items(),key=lambda x: x[1], reverse=True)}


# In[28]:


importances


# In[29]:


param_grid= {
    'n_estimators':[50,100,250],
    'max_depth':[5,10,30,None],
    'min_samples_split': [2,4],
    'max_features': ['sqrt','log2']
}
grid_search=GridSearchCV(estimator= RandomForestClassifier(),
                        param_grid = param_grid, verbose=10)


# In[30]:


grid_search.fit(train_X,train_y)


# In[31]:


RandomForest= grid_search.best_estimator_


# In[32]:


RandomForest.score(test_X, test_y)


# In[33]:


importances = dict(zip(RandomForest.feature_names_in_, RandomForest.feature_importances_))
importances= {k: v for k, v in sorted(importances.items(),key=lambda x: x[1], reverse=True)}


# In[34]:


importances


# In[ ]:




