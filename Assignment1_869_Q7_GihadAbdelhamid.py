#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Gihad Abdelhamid
# 20196899
# MMA
# W21
# 869
# 16-Aug-2020

# Answer to Question 7, Part 7a
#Load, clean, and preprocess the data as you find necessary


# ## 7) 2a - Preprocess the data however you see fit. In code comments, describe what you did and why.

# In[72]:


#Import Packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[76]:


# Read in data from OJ.csv I am assuming that you have the file in the same directory as the py file
df = pd.read_csv("OJ.csv")


# In[77]:


# Check some data frame info
df.info()

# Create variables to identify the ID column and the target
Id_col = 'Unnamed: 0'
target_col = 'Purchase'


# In[78]:


# Convert some features into categories since they are not really numeric 
# and will result in inaccuracy issues if they are kept as numeric
df['Purchase']=df['Purchase'].astype('category')
df['StoreID']=df['StoreID'].astype('category')
df['SpecialCH']=df['SpecialCH'].astype('category')
df['SpecialMM']=df['SpecialMM'].astype('category')
df['STORE']=df['STORE'].astype('category')


# In[79]:


# Convert Store7 Yes/No into 1/0 flag
df['Store7'] = df.Store7.apply(lambda x: 1 if x =='Yes' else 0)
df['Store7']=df['Store7'].astype('category')


# In[80]:


# Check some data frame info after converting to categorical
df.info()


# ## 7) 2.b Splitting the data

# In[81]:


# split the data into 80/20 so that we can do cross validation after training the model on real data, 
# which is the remaining 20%

# I used the train_test_split function to create 4 data frames, training frames for the features and the target 
# and validation frames for the features and target

# The test size parameter defines the split, in this case I used 0.2 since it's an 80/20 split. 


# In[82]:


from sklearn.model_selection import train_test_split

X = df.drop([Id_col, target_col], axis=1)
y = df[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[83]:


# Inspect the dataframes 
X.info()
X.shape
X.head()

X_train.info()
X_train.shape
X_train.head()


# ## 7) 2.c Building 3 models

# ### Naive Bayes

# In[86]:


X = df.drop([Id_col, target_col], axis=1)
y = df[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes algorithm and check the performance metrics
# Naive Bayes doesn't have hyperparameters to tune
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
gnb

y_pred_gnb = gnb.predict(X_val)


# In[87]:


# Will create and print a confusion matrix to inspect the performance metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

confusion_matrix(y_val, y_pred_gnb)
print(classification_report(y_val, y_pred_gnb))


# # SVM

# In[99]:


# running with different regularization value since this is the main hyper parameter for SVM
from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=0.01)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_val)

confusion_matrix(y_val, y_pred_svm)
print(classification_report(y_val, y_pred_svm))


# In[100]:


from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=0.1)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_val)

confusion_matrix(y_val, y_pred_svm)
print(classification_report(y_val, y_pred_svm))


# In[96]:


from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_val)

confusion_matrix(y_val, y_pred_svm)
print(classification_report(y_val, y_pred_svm))


# In[97]:


from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=5)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_val)

confusion_matrix(y_val, y_pred_svm)
print(classification_report(y_val, y_pred_svm))


# In[98]:


from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=15)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_val)

confusion_matrix(y_val, y_pred_svm)
print(classification_report(y_val, y_pred_svm))


# In[101]:


# Model chosen after trying different values for the hyper parameter. I selected this based on accuracy and precision and recall
from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_val)

confusion_matrix(y_val, y_pred_svm)
print(classification_report(y_val, y_pred_svm))


# # XGBoost

# In[108]:


# XGBoost has multiple hyper parameters
# n_estimators controls the number of boosting stages
# loss function and will leave that to default
# learning_rate shrinks the contribution of each tree
# max_depth limits the maximum depth of the trees
# max_features which decideds how many features to use and whether we use sqrt or log or an interaction
#random_State just uses a fixed seed

from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=1, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[109]:


# changing the values of the max_depth hyper parameter
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[110]:


# changing the values of the max_depth hyper parameter
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=5, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[111]:


# changing the values of the n_estimators hyper parameter
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=1, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[112]:


# changing the values of the n_estimators hyper parameter
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=50, learning_rate=0.05, max_depth=1, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[114]:


# changing the values of the learning_rate hyper parameter
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=1, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[115]:


# changing the values of the learning_rate hyper parameter
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.3, max_depth=1, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[116]:


#### Chosen model after trying different hyper parameter values
from sklearn.ensemble import GradientBoostingClassifier

clf_gt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=1, 
    random_state=0)
clf_gt.fit(X_train, y_train)

y_pred_gt = clf_gt.predict(X_val)

confusion_matrix(y_val, y_pred_gt)
print(classification_report(y_val, y_pred_gt))


# In[ ]:




