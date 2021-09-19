# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 00:34:25 2021

@author: pc
"""

import pandas as pd
import matplotlib.pyplot as plt


#reading the data from your file
data=pd.read_csv('advertising.csv')
data.head()

fig , axs=plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

feature_cols=['TV']
X=data[feature_cols]
y=data.Sales


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)



result=6.97+0.0554*50
print(result)


X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds=lr.predict(X_new)
preds


data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='blue',linewidth=1)

import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales~ TV',data=data).fit()
lm.conf_int()

lm.pvalues


lm.rsquared

feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

lr=LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()


lm=smf.ols(formula='Sales ~ TV+Radio',data=data).fit()
lm.conf_int()
lm.summary()
