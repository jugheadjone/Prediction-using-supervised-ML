#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Import all the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#Reading data
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported succesfully")
s_data.head(10)


# In[14]:


#Plotting the distribution of scores
s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours v Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.show()


# In[64]:


X=s_data.iloc[:,:-1].values
y=s_data.iloc[:,1].values


# In[20]:


#Splting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[22]:


#Training the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("Training complete")


# In[27]:


#plotting the regression line
line=regressor.coef_*X+regressor.intercept_

#plotting the graph
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# #making predictions
# 

# In[29]:


print(X_test)
y_pred=(regressor.predict(X_test))


# In[38]:


#Comparing actual vs predicted
df=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
df


# In[66]:


#making 0wn prediction
hours=[[9.25]]
ownpred=(regressor.predict(hours))
print("NO of hours={}".format(hours[0][0]))
print("predicted score={}".format(ownpred[0]))


# In[ ]:


#Evaluating model
from sklearn import metrics
print("Absolute mean error:",metrics.mean_absolute_error)

