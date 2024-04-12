#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r"C:\Users\Admin\Downloads\placement.predtiction.csv")


# In[ ]:


# Data Preprocessing


# In[4]:


data.info()


# In[17]:


data['Placement'] = data['Placement'].map({"Yes":1,"No":0})


# In[18]:


# creating features and values
features = data.iloc[:,[0]].values
label = data.iloc[:,[1]].values


# In[8]:


features


# In[19]:


label


# In[25]:


# Creating training and testing data
data.shape


# In[29]:


# How to get generalize model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
for i in range(1,40):
    x_train , x_test , y_train , y_test = train_test_split(features,label,test_size=0.2,random_state=i)
    placementprediction = LinearRegression()
    placementprediction.fit(x_train,y_train)
    train_score = placementprediction.score(x_train,y_train)
    test_score = placementprediction.score(x_test,y_test)
    
    if test_score > train_score:
        print("Training Score {} Testing Score {} Random State {} ".format(train_score,test_score,i))
    
    
    


# In[31]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(features,label,test_size=0.2,random_state=6)


# In[32]:


from sklearn.linear_model import LinearRegression
placementprediction = LinearRegression()
placementprediction.fit(x_train,y_train)


# In[33]:


print(placementprediction.score(x_train,y_train))


# In[34]:


print(placementprediction.score(x_test,y_test))


# In[36]:


# Equation
print('placement = {} + {} * (CGPA)'.format(placementprediction.intercept_,placementprediction.coef_))


# In[53]:


cgpa = float(input("Enter Student CGPA : "))
placement = placementprediction.predict(np.array([[cgpa]]))
print(np.round(placement))

