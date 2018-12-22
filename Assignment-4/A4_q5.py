
# coding: utf-8

# In[1]:


# Importing libraries

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from prettytable import PrettyTable

import numpy as np
import math


# In[2]:


def compute_posterior(X, sigma_square, a, b_square):
    n = len(X)
    x_bar = (np.sum(X) * 1.0)/n
    se_square = (sigma_square * 1.0)/n
    
    x = (b_square*x_bar + se_square*a)/(b_square + se_square)
    y_square = (b_square*se_square)/(b_square + se_square)
    
    return x, y_square;


# In[3]:


def print_table(table, labels):
    t = PrettyTable(labels)
    for row in table:
        t.add_row(row)
    print(t)


# In[4]:


# Question 5(a)
sample_input = np.loadtxt("q5_sigma3.csv", delimiter=',')

variance = 9
prior_mean = 0
prior_variance = 1

table = np.empty((0,2))
for X in sample_input:
    prior_mean, prior_variance = compute_posterior(X, variance, prior_mean, prior_variance)
    table = np.vstack((table, [prior_mean, prior_variance]))


# In[5]:


print_table(table, ['Prior Mean', 'Prior Variance'])


# In[6]:


plt.figure(figsize=(14,8))
for i in range(len(table)):
    mu = table[i][0]
    variance = table[i][1]
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma), label = "Row - " + str(i+1))

plt.xlabel("X")
plt.ylabel("PDF")
plt.legend(loc = 'upper left')
plt.title('q5_a')
plt.show()


# In[7]:


# Question 5(b)
sample_input = np.loadtxt("q5_sigma100.csv", delimiter=',')

variance = 10000
prior_mean = 0
prior_variance = 1

table = np.empty((0,2))
for X in sample_input:
    prior_mean, prior_variance = compute_posterior(X, variance, prior_mean, prior_variance)
    table = np.vstack((table, [prior_mean, prior_variance]))


# In[8]:


print_table(table, ['Prior Mean', 'Prior Variance'])


# In[9]:


plt.figure(figsize=(14,8))
for i in range(len(table)):
    mu = table[i][0]
    variance = table[i][1]
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma), label = "Row - " + str(i+1))

plt.xlabel("X")
plt.ylabel("PDF")
plt.legend(loc = 'upper left')
plt.title('q5_b')
plt.show()

