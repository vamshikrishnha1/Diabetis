#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Let's start with importing necessary libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read the data file
data = pd.read_csv("diabetes.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


data.isnull().sum()


# # We can see there few data for columns Glucose , Insulin, skin thickenss, BMI and Blood Pressure which have value as 0. That's not possible,right? you can do a quick search to see that one cannot have 0 values for these. Let's deal with that. we can either remove such data or simply replace it with their respective mean values. Let's do the latter.

# In[5]:


#here few misconception is there lke BMI can not be zero, BP can't be zero, glucose, insuline can't be zero so lets try to fix it
# now replacing zero values with the mean of the column
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())


# In[6]:


#now we have dealt with the 0 values and data looks better. But, there still are outliers present in some columns.lets visualize it
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=data, width= 0.5,ax=ax,  fliersize=3)


# In[7]:


data.head()


# In[8]:


#segregate the dependent and independent variable
X = data.drop(columns = ['Outcome'])
y = data['Outcome']


# In[9]:



# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train.shape, X_test.shape


# In[10]:


import pickle
##standard Scaling- Standardization
def scaler_standard(X_train, X_test):
    #scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #saving the model
    file = open('standardScalar1.pkl','wb')
    pickle.dump(scaler,file)
    file.close()
    
    return X_train_scaled, X_test_scaled


# In[11]:


X_train_scaled, X_test_scaled = scaler_standard(X_train, X_test)


# In[12]:


X_train_scaled


# In[13]:


log_reg = LogisticRegression()

log_reg.fit(X_train_scaled,y_train)


# In[14]:


## Hyperparameter Tuning
## GridSearch CV
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# parameter grid
parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}


# In[15]:


logreg = LogisticRegression()
clf = GridSearchCV(logreg,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=10)                     # number of folds

clf.fit(X_train_scaled,y_train)


# In[16]:


clf.best_params_


# In[17]:


clf.best_score_


# ## let's see how well our model performs on the test data set.

# In[18]:


y_pred = clf.predict(X_test_scaled)


# ## accuracy = accuracy_score(y_test,y_pred) accuracy

# In[19]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[20]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[21]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[22]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[23]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[24]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[25]:


import pickle
file = open('modelForPrediction1.pkl','wb')
pickle.dump(log_reg,file)
file.close()


# In[ ]:




