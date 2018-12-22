
# coding: utf-8

# In[91]:


# Importing libraries

import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
from collections import defaultdict
import math


# In[78]:


def calcProb(sample_input):
    sample_input.sort()
    
    # Calculating frequency
    pdf = defaultdict(float)
    for x in sample_input:
        pdf[x] += 1
    
    sumTillNow = 0;
    cdf = defaultdict(float)
    
    # Calculating PDF and CDF estimates
    for key, value in pdf.items():
        pdf[key] = (value * 1.0)/len(sample_input)
        sumTillNow = pdf[key] + sumTillNow;
        cdf[key] = sumTillNow;
    
    return pdf, cdf;


# In[233]:


# Question 7a
def ques7a(sample_input):
    pdf, cdf = calcProb(sample_input);
    
    # Plotting input vs f_hat
    X = list(map(float, list(cdf.keys())))
    Y = list(map(float, list(cdf.values())))
    plt.step(X, Y)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Estimated CDF', fontsize=12)
    plt.title('Estimated CDF for ' + str(len(sample_input)) + ' samples')
    plt.grid()
    plt.show()


# In[234]:


ques7a([1.2, 3.4, 4.5, 0.8, 10.0])


# In[235]:


def ques7b(nsamp):
    n = 199
    p = 0.5
    sample_1 = np.random.binomial(n, p, nsamp)
    
    ques7a(sample_1)


# In[236]:


ques7b(10)


# In[237]:


ques7b(100)


# In[238]:


ques7b(1000)


# In[239]:


ques7b(10000)


# In[240]:


ques7b(100000)


# In[241]:


def get_cdf(cdf, key):
    val = 0;
    for k in cdf:
        if k <= key:
            val = cdf[k];
    
    return val;

def ques7c(sample_inputs):
    
    cdfList = [];
    for sample_input in sample_inputs:
        pdf, cdf = calcProb(sample_input)
        cdfList.append(cdf)
    
    uniqueInputs = np.unique(sample_inputs);
    
    finalCdf = defaultdict(float);
    for uniqueInput in uniqueInputs:
        for cdf in cdfList:
            finalCdf[uniqueInput] = finalCdf[uniqueInput] + get_cdf(cdf, uniqueInput)
        finalCdf[uniqueInput] = (1.0 * finalCdf[uniqueInput])/len(cdfList);
    
    X = list(map(float, list(finalCdf.keys())))
    Y = list(map(float, list(finalCdf.values())))
    plt.step(X, Y)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Estimated CDF', fontsize=12)
    plt.title('Estimated CDF for ' + str(len(sample_inputs)) + " students")
    plt.grid()
    plt.show()         


# In[242]:


ques7c([[1,2,2,1], [2,3,4,5], [1,2,3,1]])


# In[243]:


def ques7d(nsamp, m):
    n = 199
    p = 0.5
    
    sample_inputs = []
    for i in range(m):
        sample_inputs.append(np.random.binomial(n, p, nsamp))
    
    ques7c(sample_inputs)


# In[244]:


ques7d(10, 10)


# In[245]:


ques7d(10, 100)


# In[246]:


ques7d(10, 1000)


# In[247]:


def ques8a(sample_input):
    n = len(sample_input)
    pdf, cdf = calcProb(sample_input);
    
    normal_f_hat_ub = []
    normal_f_hat_lb = []
    
    dkw_f_hat_ub = []
    dkw_f_hat_lb = []
    
    epsilon = math.sqrt((np.log(2/0.95)/(2*n)))
    for x in cdf:
        normal_f_hat_ub.append(cdf[x]+(1.96*(math.sqrt(round(cdf[x] * ( 1 - cdf[x]), 6) / n))))
        normal_f_hat_lb.append(cdf[x]-(1.96*(math.sqrt(round(cdf[x] * ( 1 - cdf[x]), 6) / n))))
        dkw_f_hat_ub.append(cdf[x] + epsilon)
        dkw_f_hat_lb.append(cdf[x] - epsilon)

    fig, ax = plt.subplots()
    X = list(map(float, list(cdf.keys())))
    ax.step(X, normal_f_hat_ub, label = 'Normal UB')
    ax.step(X, normal_f_hat_lb, label = 'Normal LB')
    ax.step(X, dkw_f_hat_ub, label = 'DKW UB')
    ax.step(X, dkw_f_hat_lb, label = 'DKW LB')
    ax.step(X, list(cdf.values()), label = 'Estimated CDF')
    legend = ax.legend(loc = 'upper left', fontsize = 'medium')

    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Estimated CDF', fontsize=12)
    plt.title('Estimated CDF')
    plt.grid()
    plt.show()


# In[248]:


sample_input = [];
with open("q8.csv") as f:
    sample_input = [row.split()[0] for row in f]

ques8a(sample_input)

