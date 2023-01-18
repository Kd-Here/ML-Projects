#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# # Import Data

# In[12]:


X,Y = load_coffee_data();
print("Checking a random value of X whose size is 200 with 2 values x0, x1: \n", X[10])
print("Checking a random value of Y whose size is 200 with 1 values y0: \n",Y[10]) # seeing random data from given X[10][0] is giving us temp X[10][1] is duration
print(X.shape, Y.shape)


# In[13]:


plt_roast(X,Y)


# # Normalize input Data
# 

# In[30]:


norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X) 

# print(norm_l(X))

Xn = norm_l(X)


# Since we have 200 training data set which is small for NN so we increase the dataset using tile method in numpy
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print("See the difference before tile and after tile.\n")
print("Before tile :- ",Xn.shape,Y.shape)

print("After tile :-",Xt.shape, Yt.shape)   


# # Model choosing 
# - Since doing a temp and time prediction we using simple NN of tensorflow 

# In[38]:


model = Sequential(
    [
        tf.keras.Input(shape=(2,)),  #try without this get's an error bcoz we need to build model first then we can see summary
        Dense(3, activation='sigmoid', name = '1stlayer'),
        Dense(1, activation='sigmoid', name = '2ndlayer')
     ]
)
model.summary()


# In[39]:


model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,
)


# # Predictions

# In[40]:


X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test) #step of normalizing testing data
predictions = model.predict(X_testn)
print("predictions = \n", predictions)


# In[41]:


yhat = np.zeros_like(predictions) #create zero matrix same as predictions
print(yhat) #creating yhat same as our predictions otp
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")


# In[ ]:




