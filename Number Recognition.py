#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from keras.preprocessing.text import Tokenizer as token
import tensorflow as tensorflow
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from string import punctuation
import re
import pickle as p
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


train_df = pd.read_csv('train.csv')
train_df


# In[3]:


import pandas as pd

# Example data for a DataFrame
data = {'Column1': [1, 2, 3, 4, 5],
        'Column2': ['A', 'B', 'C', 'D', 'E']}

# Create a DataFrame
train_df = pd.DataFrame(data)

# Now you can use train_df.describe()
train_df.describe()


# In[5]:


target = train_df['label']
train_df = train_df.drop(columns='label')


# In[6]:


green = '#52BE80'
red = '#EC7063'
sns.countplot(x=target, palette=[green, red])


# In[7]:


X_train, X_val, y_train, y_val = train_test_split(train_df, target, test_size=0.2, random_state=64)
print('Shape of train', X_train.shape)
print('Shape of Validation ', X_val.shape)


# In[9]:


from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest Classifier with the specified settings
rfc = RandomForestClassifier(criterion='entropy', n_estimators=700)

# Now you can use the rfc object for your machine learning tasks


# In[10]:


rfc.fit(X_train, y_train)


# In[16]:


import pickle as p  # Import the pickle module

# Assuming you've already created and trained the rfc model

filemodelname = 'rfc'  # The name of the file where you want to save the model

# Save the trained model to a file
with open(filemodelname, 'wb') as handle:
    p.dump(rfc, handle, protocol=p.HIGHEST_PROTOCOL)


# In[ ]:


def predict(inpt, model):
    with open(model, 'rb') as handle:
        model = p.load(handle)
        inn = []            
        val_pred = model.predict(inpt)
        arr = np.array(val_pred)
        unique, counts = np.unique(arr, return_counts=True)
        val_pred = dict(zip(unique, counts))
        print(max(val_pred))
        #return(val_pred)
    #try:
    #except:
        #return 'The exeption is in RandomForest.predict'


# In[13]:


test_df = pd.read_csv('test.csv')
test_df


# In[15]:


X_train, X_val = train_test_split(test_df, test_size=0.2, random_state=64)
X_train


# In[12]:


X_train


# In[18]:


# Assuming you've already created and trained the rfc model

# inpt is your input data for which you want to make predictions
predictions = rfc.predict(inpt)

# 'predictions' now contains the predicted labels for the input data 'inpt'


# In[ ]:




