#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:29:23 2019

@author: casper
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#def pearsonr(x, y):
  

#Importing data
data = pd.read_csv('prostate.csv')


data = data.iloc[:,1:10]    # First column was index colum so it was removed
data_val = data.iloc[:,0:9].values      # Values of data were extracted from pandas dataframe
col = data.columns      # Name of each variable was extracted, it will be used to name colums and indexes late in the code


#Standerdizing data with mean = 0, variance = 1
for i in range(9):
    var = data.iloc[:,i]
    m_v = np.sum(var)/len(var)
    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
    data_val[:,i] = (var-m_v)/s_v       #Data is overwritten in "data_val" variable

data = pd.DataFrame(data_val)   # Standerdized data was converted into pandas dataframe
data.columns = [col[0:9]]       # Columns were named

#Apply Praeson Correlation to find correlation within input variables and with output variable
out_correlation = np.zeros((9,9))
for i in range(9):
    first = data.iloc[:,i]
    m_f = np.sum(first)/len(first)
    for j in range(9):
        second = data.iloc[:,j]
        m_s = np.sum(second)/len(second)
        num = np.sum(np.multiply((first-m_f),(second-m_s)))
        den = np.sqrt(np.multiply(np.sum((first-m_f)**2),np.sum((second-m_s)**2)))
        out_correlation[i,j] = num/den

# Conversion into pandas dataframe
out_correlation = pd.DataFrame(out_correlation)
out_correlation.columns = [col[0:9]]
out_correlation.index = [col[0:9]]

#Showing results of correlation
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(out_correlation,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()


########################################################################

#Extracting input and output variables from data
X = data.iloc[:,0:8].values
y = data.iloc[:,8].values
# Spliting data set into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


newB = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train)), X_train.T), y_train)

##Initializing variables
#v_len = len(y_train)    # Length of data
##x0 = np.ones(v_len)
##X_train = np.concatenate((x0[:, None], X_train), axis=1)    #padding a colum of ones in data for b0
#X_train.T
#b = np.zeros(8) #initializing coeeficient values
#alpha = 0.001   # Initializing learning rate
#
## Defining cost function to find cost after each iteration
#def cost_function(X, Y, B):
#    m = len(Y)
#    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
#    return J
#
## Initial cost was found
#inital_cost = cost_function(X_train, y_train, b)
#print(inital_cost)
#
## Gradient descent function was used to minimize the cost
#def gradient_descent(X, Y, B, alpha, iterations):
#    cost_history = [0] * iterations
#    m = len(Y)
#    
#    for iteration in range(iterations):
#        # Hypothesis Values
#        h = X.dot(B)
#        # Difference b/w Hypothesis and Actual Y
#        loss = h - Y
#        # Gradient Calculation
#        gradient = X.T.dot(loss) / m
#        # Changing Values of B using Gradient
#        B = B - alpha * gradient
#        # New Cost Value
#        cost = cost_function(X, Y, B)
#        cost_history[iteration] = cost
#        
#    return B, cost_history
#
## New value of coeeficient was found
#newB, cost_history = gradient_descent(X_train, y_train, b, alpha, 10000)

# predict output values using test input data
#x0 = np.ones(len(y_test))
#X_test = np.concatenate((x0[:, None], X_test), axis=1)
Y_pred = X_test.dot(newB)

# Standerd error was found
std_err = np.zeros(8)
f = np.sum((y_test-Y_pred)**2)
#std_err[0] = np.sqrt(f/len(y_test))
for i in range(8):
    s = np.sum((X_test[:,i]-(np.sum(X_test[:,i]))/len(X_test[:,i]))**2)
    std_err[i] = np.sqrt(f/(s*(len(y_test-2))))

# Z-Values
Z_value = newB/std_err

y_diff = y_test-Y_pred
mpe = np.sum(y_diff)/len(y_diff)

# Naming indexes and columns of output data
newB = pd.DataFrame(newB)
newB.index = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
newB.columns = ['Coeff']

std_err = pd.DataFrame(std_err)
std_err.index = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
std_err.columns = ['Std Err']

Z_value = pd.DataFrame(Z_value)
Z_value.index = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
Z_value.columns = ['Z Score']

#mpe = pd.DataFrame(mpe)
#mpe.index = ['with all', 'age', 'lbph', 'lcp', 'gleason', 'pgg45', 'witout all']
#plt.plot(mpe)
