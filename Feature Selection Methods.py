#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# # Recursive Feature Elimination

# In[3]:


data = pd.read_csv(r"C:\Users\Admin\Desktop\Machine Learning (Data Sets)\50_Startups.csv")


# In[4]:


features = data.iloc[:,:-1].values
label = data.iloc[:,[4]].values


# In[5]:


from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
stateencoder = LabelEncoder()
features[:,3] = stateencoder.fit_transform(features[:,3])
ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse_output=False), [3]),],  
    remainder='passthrough' 
)
features = ct.fit_transform(features)


# In[6]:


features


# In[8]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[9]:


from sklearn.feature_selection import RFE
selectfeatures = RFE(estimator=model,step=1)
selectfeatures.fit(features,label)
print(selectfeatures.support_)


# # Feature By Model

# In[11]:


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()


# In[13]:


from sklearn.feature_selection import SelectFromModel
selectfeatures1 = SelectFromModel(model)
selectfeatures1.fit(features,label)
print(selectfeatures1.get_support())


# # Annova

# In[14]:


from sklearn.linear_model import LinearRegression
model2 = LinearRegression()


# In[15]:


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
selectfeatures2 = SelectPercentile(percentile=50,score_func = f_regression)
selectfeatures2.fit(features,label)
print(selectfeatures1.get_support())


# In[ ]:




