#!/usr/bin/env python
# coding: utf-8

# In[207]:


import pandas as pd
import numpy as np


# In[208]:


data = pd.read_csv(r"C:\Users\Admin\Downloads\Cardetails.csv")


# In[209]:


data


# In[210]:


data.columns


# In[211]:


data.info()


# In[212]:


# Droping Columns That are not necessary ex - Torque
data.drop(columns='torque',axis=1,inplace=True)


# In[213]:


data.head(2)


# In[214]:


# Data Preprocesing
# Dealing wiht null values

data.isnull().sum()


# In[215]:


data.dropna(axis='index',how='any',inplace=True)


# In[216]:


data.isnull().sum()


# In[14]:


# Duplicates Check


# In[217]:


data.duplicated().sum()


# In[218]:


data.drop_duplicates(inplace=True)


# In[219]:


data.duplicated().sum()


# In[220]:


data.describe()


# In[221]:


data.info()


# In[222]:


# Data Analysis
# Extracting Brand Name from name columns


# In[223]:


data.head(2)


# In[226]:


def get_brand_name(car_name):
    car_name = car_name.split(" ")[0]
    return car_name  


# In[227]:


data['name'] = data['name'].apply(get_brand_name)


# In[228]:


data.head()


# In[229]:


def get_mileage(mileage):
    mileage = mileage.split(" ")[0]
    return mileage


# In[230]:


data['mileage'] = data['mileage'].apply(get_mileage)


# In[231]:


data.head()


# In[232]:


def get_engine(engine):
    engine = engine.split(" ")[0]
    return engine


# In[233]:


data['engine'] = data['engine'].apply(get_engine)


# In[234]:


data.head()


# In[235]:


def get_power(power):
    power = power.split(" ")[0]
    return power


# In[236]:


data['max_power'] = data['max_power'].apply(get_power)


# In[237]:


data


# In[238]:


data.info()


# In[239]:


data['mileage'] = data['mileage'].astype(float)


# In[240]:


data['engine'] = data['engine'].astype(float)


# In[241]:


data['max_power'].replace(" ",0)


# In[242]:


data['name'].nunique()


# In[243]:


data = data[data['max_power'].notna() & data['max_power'].str.isdigit()]


# In[244]:


data['max_power'] = data['max_power'].astype(float)


# In[245]:


data.info()


# In[249]:


# Label Encoding on categorical columns


# In[250]:


data.info()


# In[254]:


from sklearn.preprocessing import LabelEncoder
data[['name','fuel','seller_type','transmission','owner']] = data[['name','fuel','seller_type','transmission','owner']].apply(LabelEncoder().fit_transform)


# In[255]:


data


# In[256]:


# Creating Features and Label For model

features = data.iloc[:,[0,1,3,4,5,6,7,8,9,10,11]].values
label = data.iloc[:,[2]].values


# In[257]:


features.ndim


# In[258]:


label.ndim


# In[259]:


features


# In[260]:


label


# In[262]:


data.info()


# In[292]:


# Creating training and testing data
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train , y_test = train_test_split(features,label,test_size=0.2,random_state=732)


# In[277]:


x_train.shape


# In[278]:


y_train.shape


# In[293]:


# Model Creation
from sklearn.linear_model import LinearRegression
priceprediction = LinearRegression()
priceprediction.fit(x_train,y_train)


# In[294]:


#Model Generalization
print(priceprediction.score(x_train,y_train))


# In[295]:


print(priceprediction.score(x_test,y_test))


# In[296]:


data.info()


# In[300]:


data.head(2)


# In[302]:


# Model Deployement
name = int(input("Enter the car : "))
year = int(input("Enter the year : "))
km_driven = int(input("Enter the km driven : "))
fuel = int(input("Enter fuel type : "))
seller = int(input("Enter the Seller type : "))
transmission = int(input("Enter the transmission : "))
owner  = int(input("Enter the Owner : "))
mileage = float(input("Enter the mileage : "))
engine = float(input("Entert the engine : "))
max_power = float(input("Ente the Max Power : "))
seats = float(input("Enter the seats : "))
features = np.array([[name,year,km_driven,fuel,seller,transmission,owner,mileage,engine,max_power,seats]])
profit = priceprediction.predict(features)
print(profit)


# In[305]:


import pickle as pk
pk.dump(priceprediction,open("price.pk",'wb'))


# In[ ]:




