#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np


# In[67]:


data = pd.read_csv(r"C:\Machine Learning\Dataset\USA_Housing.csv")


# In[68]:


# Removing less important columns


# In[69]:


data.drop(columns="Address",axis=1,inplace=True)


# In[70]:


# Data Preprocesing
# Hanling missing data
data.isnull().sum()


# In[71]:


# Duplicate Data
data.duplicated().sum()


# In[72]:


data.info()


# In[73]:


# Creating features and Columns

features = data.iloc[:,[0,1,2,3,4]].values
label = data.iloc[:,[5]].values


# In[74]:


features


# In[75]:


label


# # Feature selection using correlation

# In[76]:


data.corr()


# In[77]:


# As per the correlation we can take 0,1,2,4 as our features for our model

features = data.iloc[:,[0,1,2,4]].values
label = data.iloc[:,[5]].values


# In[78]:


# Creating Training and Testing set

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(features,label,test_size=0.2,random_state=1)


# In[79]:


# Creating Model

from sklearn.linear_model import LinearRegression

priceprediction = LinearRegression()
priceprediction.fit(x_train,y_train)


# In[80]:


# Generalization

print(priceprediction.score(x_train,y_train))


# In[81]:


print(priceprediction.score(x_test,y_test))


# In[82]:


data.info()


# In[83]:


# Model Deployment
income = float(input("Enter the income : "))
house_age = float(input("Enter the house_age : "))
rooms = float(input("Enter the number of rooms : "))
population = float(input("Enter the Area Population : "))
features = np.array([[income,house_age,rooms,population]])
profit = priceprediction.predict(features)
print(profit)


# # Feature Selection using Recursive feature Elimination

# In[84]:


# Creating features and Columns

features1 = data.iloc[:,[0,1,2,3,4]].values
label1 = data.iloc[:,[5]].values


# In[85]:


data.info()


# In[86]:


features1


# In[87]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[88]:


from sklearn.feature_selection import RFE
selectfeature = RFE(estimator=model,step=1)
selectfeature.fit(features1,label)
print(selectfeature.support_)


# In[89]:


data.info()


# In[90]:


# As per recursive feature selection we have to take only 2 feature 


# In[91]:


features2 = data.iloc[:,[1,2]].values
label2 = data.iloc[:,[5]].values


# In[92]:


# Creating Training and Testing set

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(features2,label2,test_size=0.2,random_state=1)


# In[93]:


# Creating Model

from sklearn.linear_model import LinearRegression

priceprediction = LinearRegression()
priceprediction.fit(x_train,y_train)


# In[94]:


# Generalization

print(priceprediction.score(x_train,y_train))


# In[95]:


print(priceprediction.score(x_test,y_test))


# In[96]:


data[1:2]


# In[98]:


# Model Deployment
house_age = float(input("Enter the house_age : "))
rooms = float(input("Enter the number of rooms : "))
features2 = np.array([[house_age,rooms]])
profit = priceprediction.predict(features2)
print(profit)


# # Feature Selection using Featue by model

# In[99]:


# Creating features and Columns

features3 = data.iloc[:,[0,1,2,3,4]].values
label3 = data.iloc[:,[5]].values


# In[100]:


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()


# In[101]:


from sklearn.feature_selection import SelectFromModel
selectfeature2 = SelectFromModel(model1)
selectfeature2.fit(features3,label)
print(selectfeature2.get_support())


# In[ ]:


#As this also sujjest us to use only 2 features for prediction then the score will remain same


# # Feature Selection using ANNOVA 

# In[102]:


# Creating features and Columns

features4 = data.iloc[:,[0,1,2,3,4]].values
label4 = data.iloc[:,[5]].values


# In[103]:


from sklearn.linear_model import LinearRegression
model2 = LinearRegression()


# In[104]:


from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
selectfeatures3 = SelectPercentile(percentile=50,score_func=f_regression)
selectfeatures3.fit(features4,label4)
print(selectfeatures3.get_support())


# In[ ]:


#As this also sujjest us to use only 2 features but the diiferent one


# In[105]:


data.info()


# In[106]:


#Creating Features for model

features5 = data.iloc[:,[0,1]].values
label5 = data.iloc[:,[5]].values


# In[107]:


# Creating Training and Testing set

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(features5,label5,test_size=0.2,random_state=1)


# In[108]:


# Creating Model

from sklearn.linear_model import LinearRegression

priceprediction = LinearRegression()
priceprediction.fit(x_train,y_train)


# In[109]:


# Generalization

print(priceprediction.score(x_train,y_train))


# In[110]:


print(priceprediction.score(x_test,y_test))


# In[111]:


data.info()


# In[112]:


# Model Deployment
income = float(input("Enter the income : "))
house_age = float(input("Enter the number of house_age : "))
features5 = np.array([[income,house_age]])
profit = priceprediction.predict(features5)
print(profit)


# # Feature Selection using Backward Elimination

# In[114]:


features6 = data.iloc[:,[0,1,2,3,4]].values
label6 = data.iloc[:,[5]].values


# In[116]:


data.info()


# In[119]:


featuresallin = np.append(np.ones((5000,1)).astype(int),features6,axis=1)


# In[120]:


featuresallin


# In[121]:


# significance level as per my choice is 0.6 means I can tolerate 6 % of error


# In[125]:


import statsmodels.api as stat
model5 = stat.OLS(endog=label6,exog=featuresallin).fit()
model5.summary()


# In[126]:


# As per Backward elimination we can use 4 features for our prediction


# In[127]:


features7 = data.iloc[:,[0,1,2,4]].values
label7 = data.iloc[:,[5]].values


# In[128]:


# Creating Training and Testing set

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(features7,label7,test_size=0.2,random_state=1)


# In[129]:


# Creating Model

from sklearn.linear_model import LinearRegression

priceprediction = LinearRegression()
priceprediction.fit(x_train,y_train)


# In[130]:


# Generalization

print(priceprediction.score(x_train,y_train))


# In[131]:


print(priceprediction.score(x_test,y_test))


# In[132]:


data.info()


# In[136]:


# Model Deployment
income = float(input("Enter the income : "))
house_age = float(input("Enter the house_age : "))
rooms = float(input("Enter the number of rooms : "))
population = float(input("Enter the Area Population : "))
features7 = np.array([[income,house_age,rooms,population]])
profit = priceprediction.predict(features7)
print(profit)


# In[134]:


data.head(4)

