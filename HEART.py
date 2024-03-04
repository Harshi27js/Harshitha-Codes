#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


# In[2]:


os.chdir("C:\\Users\\harsh\\OneDrive\\Desktop\\A1a,b")


# In[3]:


heart=pd.read_csv("heart.csv", encoding='latin1')


# In[4]:


heart.head()


# In[5]:


heart["output"].value_counts()


# In[6]:


heart.isnull().sum()


# In[7]:


heart.shape


# In[8]:


heart.columns


# In[9]:


heart.dtypes


# In[10]:


heart['oldpeak']=heart['oldpeak'].astype(int)


# In[11]:


heart.dtypes


# In[12]:


import seaborn as sns


# In[13]:


sns.boxplot(heart['trtbps'])


# In[14]:


Q1 = heart['trtbps'].quantile(0.25)
Q2 = heart['trtbps'].quantile(0.75)
iqr = Q2 - Q1
up_limit = Q2 + 1.5 * iqr
low_limit = Q1 - 1.5 * iqr


# In[15]:


heart = heart[(heart['trtbps'] <= up_limit) & (heart['trtbps'] >= low_limit)]


# In[16]:


sns.boxplot(heart['trtbps'])


# In[17]:


corrmat = heart.corr()
f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(corrmat, ax = ax, cmap = 'RdPu', linewidths = 0.1)


# In[18]:


x=heart.drop(['output'],axis=1)
y=heart['output']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=20)


# In[20]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


y_pred


# In[23]:


confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)


# In[24]:


print(classification_report(y_test, y_pred))


# In[25]:


y_pred_proba = model.predict_proba(X_test)[:,1]


# In[26]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)


# In[27]:


plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc= "lower right")
plt.show()


# In[28]:


from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree, metrics


# In[29]:


x=heart.drop(['output'],axis=1)
y=heart['output']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[31]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[32]:


y_pred =model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


# In[33]:


confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)


# In[34]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[35]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)


# In[36]:


plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc= "lower right")
plt.show()


# In[37]:





# In[ ]:




