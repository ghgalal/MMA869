#!/usr/bin/env python
# coding: utf-8

# In[417]:


# Gihad Abdelhamid
# 20196899
# MMA
# W21
# 869
# 16-Aug-2020

# Answer to Question 1, Part 1a
#Load, clean, and preprocess the data as you find necessary


# In[445]:


#Import Packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[446]:


# Read in data from jewelry_customers.csv I am assuming that you have the file in the same directory as the py file
df = pd.read_csv("jewelry_customers.csv")


# In[447]:


# Check some data frame info
df.info()


# In[448]:


# Checking if there is any negative values
df.index[df['Age'] < 0]
df.index[df['Income'] < 0]
df.index[df['SpendingScore'] < 0]
df.index[df['Savings'] < 0]


# In[449]:


# Convert the frame 
X=df.to_numpy()


# In[450]:


# Print the frame
X


# In[451]:


# Plotting Age and Income
plt.figure();
plt.scatter(X[:,0],X[:,1], c="black")
plt.xlabel('Age', fontsize=14);
plt.ylabel('Income', fontsize=14);


# In[452]:


# Plotting Age and SpendingScore
plt.figure()
plt.scatter(X[:,0],X[:,2], c="black")
plt.xlabel('Age', fontsize=14);
plt.ylabel('Spending Score', fontsize=14);


# In[453]:


# Plotting Age and Savings
plt.figure()
plt.scatter(X[:,0],X[:,3], c="black")
plt.xlabel('Age', fontsize=14);
plt.ylabel('Savings', fontsize=14);


# In[454]:


#Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled


# In[455]:


# Answer to Question 1, Part 1b
# Cluster the data using any clustering algorithm discussed in class. 
# Measure goodness-of-fit. Try different values of hyper parameters to see how they affect goodness-of-fit.
K=5

from sklearn.cluster import KMeans
k_means=KMeans(init="k-means++", n_clusters=K, random_state=42)
k_means.fit(X_scaled)

# WCSS == Inertia
print('WCSS/Intertia Score for K: {}'.format(K))
k_means.inertia_

print('Sillouette Score for K: {}'.format(K))
silhouette_score(X, k_means.labels_)
sample_silhouette_values = silhouette_samples(X, k_means.labels_)
sample_silhouette_values


# In[456]:


# Print the cluster IDs
k_means.labels_


# In[457]:


# Print the centers
k_means.cluster_centers_
means = scaler.inverse_transform(k_means.cluster_centers_)
means


# In[458]:


plt.style.use('default');

#plt.figure(figsize=(12, 10));
plt.grid(True);

sc = plt.scatter(X[:, 0], X[:, 1], s=20, c=k_means.labels_);
plt.xlabel('Age', fontsize=14);
plt.ylabel('Income', fontsize=14);
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);

plt.scatter(means[:, 0], means[:, 1], marker='x',c="black")


# In[462]:


# Playing with K, setting between 2 and 6
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, random_state=42).fit(X_scaled)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(X_scaled, kmeans.labels_)
    print("for k={}, inertia={} and silhouette={}".format(k,inertias[k], silhouettes[k]))
    

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");


plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");


# In[463]:


# Answer to Question 1, Part 1c
# Print the Stats for the whole data set
# The cluster size I selected is 5 based on the Inertia score

K=5
from sklearn.cluster import KMeans
k_means=KMeans(init="k-means++", n_clusters=K, random_state=42)
k_means.fit(X_scaled)

# Print Labels and cluster centers
k_means.labels_
k_means.cluster_centers_
means = scaler.inverse_transform(k_means.cluster_centers_)


# In[465]:


# Print the stats for each cluster
labels=k_means.labels_
for i, label in enumerate(set(labels)):
    print('\nCluster {}:'.format(label))
    print('Number of Instances: {}'.format(d.nobs))
    d = stats.describe(X_scaled[labels==label], axis=0)
    display(stats_to_df(d, scaler))


# In[466]:


from scipy import stats

# List the column names to be used in the table
col_names = ['Age','Income','Spending Score','Savings']

# Set some display options
pd.set_option("display.precision", 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Define a function to create the stats in a table format
def stats_to_df(d, scaler):
    tmp_df = pd.DataFrame(columns=col_names)
    
    tmp_df.loc[0] = scaler.inverse_transform(d.minmax[0])
    tmp_df.loc[1] = scaler.inverse_transform(d.mean)
    tmp_df.loc[2] = scaler.inverse_transform(d.minmax[1])
    tmp_df.loc[3] = scaler.inverse_transform(d.variance)
    tmp_df.loc[4] = scaler.inverse_transform(d.skewness)
    tmp_df.loc[5] = scaler.inverse_transform(d.kurtosis)
    tmp_df.index = ['Min', 'Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis'] 
    return tmp_df.T

# Print the 
print('All Data:')
print('Number of Instances: {}'.format(X.shape[0]))
d = stats.describe(X_scaled, axis=0)
display(stats_to_df(d, scaler))


# In[467]:


#trying DBSCAN
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.4, min_samples=3)
dbscan.fit(X_scaled)
dbscan.labels_
k_means.cluster_centers_
scaler.inverse_transform(k_means.cluster_centers_)

silhouettes = {}
for eps in np.arange(0.1, 0.9, 0.1):
    db = DBSCAN(eps=eps, min_samples=3).fit(X_scaled)
    silhouettes[eps] = silhouette_score(X, db.labels_, metric='euclidean')
    print("for eps={}, silhouette={}".format(eps, silhouettes[eps]))


# In[ ]:




