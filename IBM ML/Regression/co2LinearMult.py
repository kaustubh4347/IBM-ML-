# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:02:30 2020

@author: kaust
"""


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumptionCo2.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

print(df.head(9))

#taking some features
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#ploting data in graph
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


#spliting data for train and test and plotting it for some feature
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Multiple Linear Regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
xt = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
yt = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (xt, yt)
#the coefficient
print("Coefficients: ",regr.coef_)

x = np.asanyarray(train[['ENGINESIZE']])
y = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
z = np.asanyarray(train[['CO2EMISSIONS']])

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x, y, z, "-b");
