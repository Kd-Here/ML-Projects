#!/usr/bin/env python
# coding: utf-8

# # Steps to solve ML problems:-
# 1. [Collecting Data](#DataSet): As you know, machines initially learn from the data that you give them. 
# 2. Preparing the Data: After you have your data, you have to prepare it [**Normalize data to make in same range.**](#Normalize-Data)
# 3. Choosing a Model: ...
# 4. Training the Model: ...
# 5. Evaluating the Model: **testing the performance of the model on previously unseen data**
# 6. Parameter Tuning: ...
# 7. Making Predictions.

# In[9]:


# Importing data:
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# In[10]:


# Collecting data
A, B = load_coffee_data()
# A[0] is temperature and A[1] time
print(A.shape,B.shape)
print(A[139],B[139])


# In[11]:


# Preparaing data creating it of same range i.e normalize
# It's called preprocessing step in ML so check tensrorflow guide of normalization 
layer = tf.keras.layers.Normalization()
layer.adapt(A)
normalA = layer(A)
# print(normalA)
# print(A)
# print(B)
k = normalA[0:10]
print(k)
t = B[0:10]
print(t)


# ### We are choosing NN model since it's an classification prb we can have logistic regression,
# - With logistic we can use sigmoid function which we done in earlier session supervised method.
# - For NN we will use 2 layers and 3 and 1 neurons repesctively.
#  [Reffer for counting neurons number](https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e)

# In[12]:


plt_roast(A,B)


# **Since we are able to linearly separated dataset by using 3 lines our 1st layer should have 3 neurons, our 2nd layer will be classifier bcoz we can accurately separated them. So it's 2 layer two NN** 

# ### Model
#    <right> <img  src="./images/C2_W1_RoastingNetwork.PNG" width="200" />   <right/>  
# Let's build the "Coffee Roasting Network" described in lecture. There are two layers with sigmoid activations as shown below:

# In[13]:


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),  
        tf.keras.layers.Dense(3, activation='sigmoid', name = '1stlayer'),
        tf.keras.layers.Dense(1, activation='sigmoid', name = '2ndlayer')
     ]
)


# In[14]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                 optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)


# In[46]:


model.fit(
    x=k,
    y=t,
    epochs=10,

)


# In[47]:


W1, b1 = model.get_layer("1stlayer").get_weights()
W2, b2 = model.get_layer("2ndlayer").get_weights()

print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)


# In[ ]:


# W1 = np.array([
#     [-8.94,  0.29, 12.89],
#     [-0.17, -7.34, 10.79]] )
# b1 = np.array([-9.87, -9.28,  1.01])
# W2 = np.array([
#     [-31.38],
#     [-27.86],
#     [-32.79]])
# b2 = np.array([15.54])
# model.get_layer("1stlayer").set_weights([W1,b1])
# model.get_layer("2ndlayer").set_weights([W2,b2])


# In[ ]:


X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = layer(X_test) #step of normalizing testing data
predictions = model.predict(X_testn)
print("predictions = \n", predictions)


# In[ ]:


yhat = np.zeros_like(predictions)

print(yhat) #creating yhat same as our predictions otp
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")


# In[ ]:





# In[ ]:


import numpy as np
def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g


# In[22]:


W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
w1 = W1[:,0]
print(w1)
ain = A[0,:]
print(ain)
b = np.array( [-9.82, -9.28,  0.96] )
# b = np.array([-9.87])
print(b)
c = sigmoid(np.dot(w1,ain)+b)
print(sigmoid)
print(c)


# In[21]:


W = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
units = W.shape[1]
a_out = np.zeros(units)
for j in range(units):               
        w = W[:,j]
        print("w:",w)
        z = np.dot(w, A[j,:]) + b[j] 
        print("A[j,:]",A[j,:])
        print("z:",z)
        a_out[j] = sigmoid(z)  
        print("aj:",a_out[j])
print("Aout",a_out)


# In[ ]:





# In[23]:


W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
a = W2_tmp.shape
vex = np.array([1, 2, 3])
b2_tmp = np.array( [15.41] )


# ## We are considering 1 input at time for all neurons 

# In[24]:


import numpy as np
def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g

def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = sigmoid(z)  
#         print("a_out[",j,"]:",a_out)
#     print(a_out)
    return(a_out)
# print("Input is:-",A[12])
# print("Output:-")
c = my_dense(A[8],W1_tmp,b1_tmp)
# print("Finally output of all neurons:\n",c)


# ## Created dense function for 1 input with 3 neurons now we are creating layers function for handling multiple layers

# In[25]:


def seq(xin,W1,b1,W2,b2):
    print("We are starting with layer 1 and began with dense for layer 1")
    a1 = my_dense(xin,W1,b1)
    print("\n We are starting with layer 1 and began with dense for layer 2")
    a2 = my_dense(a1,W2,b2)
    return a2
for i in range(200):
    out = seq(A[i],W1_tmp,b1_tmp,W2_tmp,b2_tmp)
print(out)


# In[26]:


# import numpy as np
# def sigmoid(z):
#     """
#     Compute the sigmoid of z

#     Parameters
#     ----------
#     z : array_like
#         A scalar or numpy array of any size.

#     Returns
#     -------
#      g : array_like
#          sigmoid(z)
#     """
#     z = np.clip( z, -500, 500 )           # protect against overflow
#     g = 1.0/(1.0+np.exp(-z))

#     return g

# def my_dense(a_in, W, b):
#     units = W.shape[1]
#     a_out = np.zeros(units)
#     for j in range(units):               
#         w = W[:,j]                                    
#         z = np.dot(w, a_in) + b[j]         
#         a_out[j] = sigmoid(z)  
# #         print("a_out:",a_out)
#     return(a_out)


# def my_sequential(x, W1, b1, W2, b2):
#     a1 = my_dense(x,  W1, b1)
# #     print("A1",a1.shape)
#     a2 = my_dense(a1, W2, b2)
#     return(a2)


# def my_predict(X, W1, b1, W2, b2):
#     m = X.shape[0]
#     p = np.zeros((m,1))
# #     print(p)
#     for i in range(m):
#         p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
# #     print(p)
#     return(p)
# m = A.shape[0]
# print(m)


# In[37]:


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)


# In[42]:


predictions = my_predict(normalA, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
print(predictions)


# In[43]:


yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")


# In[ ]:





# In[ ]:




