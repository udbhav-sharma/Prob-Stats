
# coding: utf-8

# In[1]:


# Importing libraries

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math


# In[2]:


X = np.loadtxt("q7_X.csv")
Y = np.loadtxt("q7_Y.csv")


# In[3]:


# Q-7a

n = len(X)
m = len(Y)

X_bar = np.mean(X)
Y_bar = np.mean(Y)

delta = X_bar - Y_bar
se = math.sqrt((np.var(X))/n + (np.var(Y))/m)

w = (delta - 0)/se

print(w)


# In[6]:


# Q-7b

D = X-Y

n = len(X)
D_bar = sum(D)/n
sd = math.sqrt(np.var(D))

T = (D_bar * math.sqrt(n))/sd

print(T)

