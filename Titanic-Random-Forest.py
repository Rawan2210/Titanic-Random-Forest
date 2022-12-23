#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# In[16]:


data=pd.read_csv("data_titanic.csv")
data.head()


# In[17]:


data.info()


# In[18]:


data["Cabin"] = data.Cabin.astype("category").cat.codes
data["Embarked"] = data.Embarked.astype("category").cat.codes
data["Sex"] = data.Sex.astype("category").cat.codes
data["Survived"] = data.Survived.astype("category")


# In[19]:


data.info()


# In[20]:


data.isnull().sum()


# In[21]:


data["Age"].fillna(data["Age"].mean(),inplace=True)


# In[22]:


data.isnull().sum().sum()


# In[23]:


from sklearn.model_selection import train_test_split
SEED=15
x = data.drop(["Name","Ticket","Survived","PassengerId"],axis=1)
y = data["Survived"].cat.codes

#splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=SEED)


# In[24]:


from sklearn import tree   
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


model = tree.DecisionTreeClassifier(random_state=SEED)

model.fit(x_train, y_train)   
y_pred=model.predict(x_test)   

print("score:{}".format(accuracy_score(y_test, y_pred)))


# In[25]:


fig = plt.figure(figsize=(100,100))
tree.plot_tree(model, 
                   feature_names=x.columns.values.tolist(),  
                   class_names=data.Survived.unique(),
                   filled=True)
fig.savefig("tree.png")


# In[ ]:


model1 = tree.DecisionTreeClassifier(min_impurity_split=0.2,max_depth=4,random_state=SEED)  
model1.fit(x_train, y_train)   
y_pred=model1.predict(x_test)

print("score:{}".format(accuracy_score(y_test, y_pred)))


# In[ ]:


fig = plt.figure(figsize=(100,100))
tree.plot_tree(model1,
                   feature_names=x.columns.values.tolist(),  
                   class_names=data.Survived.unique(),
                   filled=True)
fig.savefig("tree1.png")

