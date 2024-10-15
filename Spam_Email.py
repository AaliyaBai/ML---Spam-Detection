#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("mail_data.csv")


# In[3]:


df.head(5)


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['Category'] = df['Category'].map({'spam':'0', 'ham':'1'})


# In[7]:


df


# In[8]:


X = df['Message']
y = df['Category']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.25, random_state = 42)


# In[10]:


y_train.shape


# In[11]:


vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


# In[12]:


X_train


# In[13]:


X_train_vector


# In[14]:


# Train Naive Bayes classifier
NV_model = MultinomialNB()
NV_model.fit(X_train_vector, y_train)


# In[15]:


# Predict and evaluate
predictions = NV_model.predict(X_test_vector)
#print(classification_report(y_test, predictions))


# In[16]:


predictions


# In[17]:


accuracy = NV_model.score(X_test_vector, y_test)


# In[18]:


accuracy


# In[19]:


df.shape


# In[20]:


input_user_email = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
input_data_features = vectorizer.transform(input_user_email)
predicted_input_data = NV_model.predict(input_data_features)


# In[21]:


predicted_input_data = predicted_input_data[0]


# In[24]:


predicted_input_data


# In[23]:


for labels in predicted_input_data:
    if labels == 1:
        print("ham")
    else:
        print("spam")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




