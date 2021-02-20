#!/usr/bin/env python
# coding: utf-8

# # Prediction Using Supervised Machine Learning
# #### -Ankita Komal 

# ### We will be using Linear Regression with Python Scikit Learn

# ### Problem Statement :  To predict the score of a student based on the study hour per day

# In[10]:


##importing the required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


##loading the students study hr dataset
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")
s_data.head(5)


# In[20]:


##checking the data frame info 
s_data.info()


# ##### The given dataset has 2 columns: hours (float) and scores (integer). There is no  missing data in this dataset 

# # 

# In[22]:


##Plotting the hours vs score distribution to identify the relationship b/w the number of hours studied and marks scored
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ##### The above distribution clearly shows that there is a direct linear relationship b/w the number of hours of study and marks scored. It means more the time spent on studing more is the marks obtained

# # 

# ## Preparing the Data

# In[25]:


## storing the independent and dependent variables separately
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[27]:


## cretaing the train and test data. We are considering the 80:20 split here
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[31]:


## Training the model
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ## Predicting the scores

# In[4]:


print(X_test)
y_pred = regressor.predict(X_test)


# In[39]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df.plot(kind = 'bar',figsize = (6,6), color=('grey','green'))
plt.show()


# ## Predicting the score for study hr = 9.25

# In[8]:


hours = np.array(9.25)
print("No of Hours = {}".format(hours))
hours = hours.reshape(-1,1)
score_pred = regressor.predict(hours) 
print("Predicted Score = {}".format(score_pred[0]))


# ## Model Evaluation

# In[9]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

