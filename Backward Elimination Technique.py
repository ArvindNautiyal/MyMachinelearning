#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np


# In[63]:


data = pd.read_csv(r"C:\Users\Admin\Desktop\Machine Learning (Data Sets)\50_Startups.csv")


# In[64]:


data


# In[65]:


features = data.iloc[:,:-1].values
label = data.iloc[:,[4]].values


# In[66]:


label.ndim


# In[67]:


from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
stateencoder = LabelEncoder()
features[:,3] = stateencoder.fit_transform(features[:,3])
ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), [3]),],  
    remainder='passthrough' 
)
features = ct.fit_transform(features)




# In[68]:


features


# In[81]:


#step 1 - forming All in
featuresallin = np.append(np.ones((50,1)).astype(float) , features , axis = 1)


# In[82]:


featuresallin


# In[47]:


# step 2 - 
# Deciding signinficnace level , generally decided by Data Scientist
# 0.0.5 ---- 5% error can be tolerated


# In[83]:


# step 3 - Perform OLS operation
# Goal is calculate the p-value/significance level of each features
import statsmodels.api as stat
model1 = stat.OLS(endog=label,exog=featuresallin).fit()
model1.summary()


# In[55]:


get_ipython().run_line_magic('pinfo', 'stat.OLS')


# In[77]:


featuresallin.ndim


# In[79]:


features.ndim


# In[80]:


label.ndim

