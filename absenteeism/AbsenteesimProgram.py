# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:10:09 2019

@author: Shyam
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Absenteeismnew.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,19].values

import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(dataset.corr(), annot=True, cmap='Reds')

from sklearn.model_selection import train_test_split
X_train, kX_test, y_train, ky_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100,criterion='mse')
#print(clf)
RF=clf.fit(X_train,y_train)
PredictorColumns=list(dataset.columns)
PredictorColumns.remove('ID')
PredictorColumns.remove('Absenteeism Time')
%matplotlib inline
feature_importances = pd.Series(RF.feature_importances_, index=PredictorColumns)
feature_importances.nlargest(10).plot(kind='barh')
fig=plt.figure(figsize=(15,5))

################
# LogisticRegression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(kX_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(ky_test, y_pred)


from sklearn.metrics import mean_absolute_error 
mean_absolute_error (ky_test,y_pred)

from sklearn.metrics import r2_score 
r2_score(ky_test, y_pred)



# LogisticRegression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(kX_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(ky_test, y_pred)


from sklearn.metrics import mean_absolute_error 
mean_absolute_error (ky_test,y_pred)

from sklearn.metrics import r2_score 
r2_score(ky_test, y_pred)
