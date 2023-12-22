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
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest


#Importing data
data = pd.read_csv('prostate.csv')


data = data.iloc[:,1:10]

#Extracting input and output variables from data
X = data.iloc[:,0:8]
y = data.iloc[:,8]

#Apply Praeson Correlation to find correlation within input variables
corr = data.corr(method ='pearson')

#Showing results of correlation
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

#data['lcavol'] =data['lcavol'].astype('category').cat.codes
#data['lweight'] =data['lweight'].astype('category').cat.codes
#data['age'] =data['age'].astype('category').cat.codes
#data['lbph'] =data['lbph'].astype('category').cat.codes
#data['svi'] =data['svi'].astype('category').cat.codes
#data['lcp'] =data['lcp'].astype('category').cat.codes
#data['gleason'] =data['gleason'].astype('category').cat.codes
#data['pgg45'] =data['pgg45'].astype('category').cat.codes
#data_tran = data.T
#data_tran.apply(zscore)

#Finding correlation between input and output variables
col = data.columns
out_corr = np.zeros((8,1))
for i in range(8):
    df = data.iloc[:,i]
    corr1 = np.corrcoef(df, data.iloc[:,8])
    out_corr[i] = corr1[1,0]

out_corr = pd.DataFrame(out_corr)
out_corr.columns = [col[8]]
out_corr.index = [col[0:8]]

# Spliting data set into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# fit linear model to dataset
results = sm.OLS(y,X).fit()
results.summary()

#Storing required data into variables
coef = results.params
z_values = results.tvalues
std_err = results.bse

####################################################################
# fit linear model to dataset
#results = sm.OLS(y,X).fit()
#results.summary()
#
##Storing required data into variables
#coef = results.params
#z_values = results.tvalues
#std_err = results.bse


####################################################################
#m_x = np.mean(X_train)
#m_y = np.mean(y_train)
#
#v_len = len(y_train)
#numer = 0
#denom = 0
#b = np.zeros(9)
#
#for j in range(8):
#    numer = np.sum((X_train.iloc[:,j] - m_x[j]) * (y_train.iloc[:] - m_y))
#    denom = np.sum((X_train.iloc[:,j] - m_x[j]) ** 2)
#    b[j+1] = numer / denom
#
#b0 = m_y
#for i in range(8):
#    b0 = b0 - (b[i+1] * m_x[i])
#b[0] = b0


