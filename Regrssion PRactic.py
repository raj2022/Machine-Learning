#!/usr/bin/env python
# coding: utf-8

# Linear Regression model is to obtain a line that best fits the data. By best fit, what is meant is that the total distance of all points from our regression line should be minimal. Often this distance of the points from our regression line is referred to as an Error though it is technically not one. We know that the straight line equation is of the form:
# 
#    y= mx+c
#    
#  where y is the Dependent Variable, x is the Independent Variable, m is the Slope of the line and c is the Coefficient (or the y-intercept).
#  
#  This equation is the basis for any Linear Regression problem and is referred to as the Hypothesis function for Linear Regression. The goal of most machine learning algorithms is to construct a model i.e. a hypothesis to estimate the dependent variable based on our independent variable(s).
#  
#  This hypothesis, maps our inputs to the output. The hypothesis for linear regression is usually presented as:
# 
# h_\theta(x) = \theta_0 + \theta_1(x)

# In[ ]:





# Cost functions are used to calculate how the model is performing. In layman’s words, cost function is the sum of all the errors. While building our ML model, our aim is to minimize the cost function.
# 

# Data importing

# In[1]:


#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv('kc_house_data.csv')
df.shape


#  VISUALISING THE DATA

# In[15]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr() , cmap='coolwarm')
plt.show()


# Matplotlib and Seashore are excellent libraries that can be used to visualize our data on various different plots.
# 

#  FEATURE ENGINEERING:
#         While visualizing our data, we found that there is a strong correlation between the two parameters: sqft_living and price. Thereby we will be using these parameters for building our model.

# In[18]:


area = df['sqft_living']
price = df['price']

x = np.array(area).reshape(-1,1)
y = np.array(price)


# In[19]:


area


# In[20]:


x


# After selecting the desired parameters the next step is to import the method train_test_split from sklearn library. This is used to split our data into training and testing data. Commonly 70–80% of the data is taken as the training dataset while the remaining data constitutes the testing dataset.

# In[21]:


#Import LinearRegression and split the data into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 0)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
#Fit the model over the training dataset
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# After this LinearRegression is imported from sklearn.model_selection and the model is fit over the training dataset. The intercept and coefficient of our model can be calculated as shown below:
# 

# In[33]:


#Calculate intercept and coefficient
print(model.intercept_)
print(model.coef_)
pred=model.predict(X_test)
predictions = pred.reshape(-1,1)
from sklearn.metrics import mean_squared_error
print('MSE : ', mean_squared_error(y_test,predictions))
print('RMSE :', np.sqrt(mean_squared_error(y_test,predictions)))


# In[ ]:


print('RMSE :', np.sqrt(mean_squared_error(y_test,predictions)))


# # Linear Regression Using Gradient Descent
# 

# Gradient descent is an iterative optimization algorithm to find the minimum of a function. To understand this algorithm imagine a person with no sense of direction who wants to get to the bottom of the valley.

# In[34]:


#Initializing the variables
m = 0
c = 0
L = 0.001
epochs = 100
n = float(len(x))


# The gradient descent approach is applied step by step to our m and c. Initially let m = 0 and c = 0. Let L be our learning rate. This controls how much the value of m changes with each step.
# 

# In[ ]:


for i in range(epochs):
    Y_pred=m*x+c
    Dm = (-2/n)*sum(x*(y-Y_pred))
    Dc = (-2/n)*sum(y-Y_pred)
    m = m-L*Dm
    c = c-L*Dc
print(m,c)
#Predicting the values
y_pred = df['sqft_living'].apply(lambda a:c+m*a)
y_pred.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




