#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression


# In[2]:


df =  pd.read_csv('diabetes_dataset.csv')
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.columns


# In[5]:


X = df.drop('Outcome', axis= 1)
X.head()


# In[6]:


y = df['Outcome']
y


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


normalize = StandardScaler()
normalize.fit(X)
X = normalize.fit_transform(X)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[10]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[14]:


y_pred = model.predict(X_test)
y_pred


# In[12]:


accuracy = accuracy_score(y_test,y_pred)
recal = recall_score(y_test,y_pred)
precesion = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[13]:


print('Accuracy : ' ,accuracy)
print('recall : ' ,recal)
print('Precesion : ' ,precesion)
print('f1 : ' ,f1)


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score


# In[32]:


data = pd.read_csv('diabetes_dataset.csv')
data.head()


# In[33]:


data.isnull().sum()


# In[34]:


data.columns


# In[35]:


X = data.drop(['Outcome'],axis = 1)
X.head()


# In[36]:


Y = data['Outcome']
Y


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


# normalize = StandardScaler()
# normalize.fit(X)
# X = normalize.fit_transform(X)


# In[37]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)


# In[38]:





# In[39]:





# In[40]:


model = LogisticRegression()
model.fit(X_train,Y_train)


# In[41]:


Y_pred = model.predict(X_test)
Y_pred


# In[42]:


accuracy = accuracy_score(Y_test,Y_pred)
f1_score = f1_score(Y_test,Y_pred)
precesion = precision_score(Y_test,Y_pred)
recal = recall_score(Y_test,Y_pred)


# In[43]:


print("The accuracy score is : " ,accuracy )
print("The f1 score is : ", f1_score)
print("The precesion score is : ", precesion)
print("The recal score is :" , recal)


# In[ ]:





# In[ ]:




