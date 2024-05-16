#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# In[19]:


data = pd.read_csv("spam.csv", encoding='latin-1')
# Drop unnecessary columns and rename columns for clarity
data = data[['v1', 'v2']]
data.columns = ['label', 'message']


# In[20]:


x = data['message']
y = data['label']


# In[21]:


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(x)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[23]:


clf = MultinomialNB()
clf.fit(X_train, y_train)


# In[25]:


sample = input('Enter a message: ')
input_data = vectorizer.transform([sample])


# In[26]:


prediction = clf.predict(input_data)
print("The message is classified as:", prediction[0])


# In[27]:


y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




