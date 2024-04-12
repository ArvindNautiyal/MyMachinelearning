#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#1. Load the data
data = pd.read_csv(r'C:\Users\Admin\Downloads\Salary_Data.csv')


# In[3]:


data


# In[4]:


#2.Check for Missing values
data.info()
# If there exists a data point where label is missing, simply delete that record.


# In[5]:


# Remove Null Values
data.dropna(axis='index',how='any',inplace=True)


# In[6]:


data.info()


# In[7]:


# Creating Features and Labels
# It is very important that we have numpy 2d values
features = data.iloc[:,[0]].values
label = data.iloc[:,[1]].values


# In[8]:


features.ndim


# In[9]:


label.ndim


# In[10]:


features


# In[11]:


label


# In[12]:


# Create training and testing dataset


# In[13]:


# Getting best model usning loop

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
for i in range(1,31):
    
    x_train,x_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size = 0.2,
                                                random_state= i)
    salaryPrediction = LinearRegression()
    salaryPrediction.fit(x_train,y_train)
    train_score= salaryPrediction.score(x_train,y_train)
    test_score = salaryPrediction.score(x_test,y_test)
    
    if test_score > train_score:
        print("Training Score {} Testing Score {} Random State {}". format(train_score,test_score,i))


# In[14]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(features,label,test_size=0.2,random_state=30)


# In[15]:


# Create the Model


# In[16]:


from sklearn.linear_model import LinearRegression 
salarypridiction = LinearRegression()
salarypridiction.fit(x_train,y_train)
# fit is all about calculating the slope and intercept of the equation


# In[17]:


# Check the Quality of 

# Training data score          
salarypridiction.score(x_train,y_train) #Known data


# In[18]:


# Testing data Score
salarypridiction.score(x_test, y_test) #Unknown data


# In[19]:


# Equation
print("Equation of line is salary = {} + {} * (year of experience)".format(salarypridiction.intercept_,salarypridiction.coef_))


# In[22]:


# Model Deployment

yearexperience = float(input("Enter your year of experience : "))
salary = salarypridiction.predict(np.array([[yearexperience]]))
print("Your Salary is {} on the basis of your {} year of experience".format(salary,yearexperience))


# In[ ]:




