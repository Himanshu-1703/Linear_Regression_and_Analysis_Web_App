import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
import statsmodels.api as sm



'''
There are a few assumptions made on the data before we fit the linear regression model on it. 
These assumptions if True cement the fact that the data is linear or 
somewhat linear in its distribution and linear models like linear regression can be applied on it. 

Applying linear regression on non linear data can result in wrong estimation 
about the values of the coef and the intercept.

Generally there are 5 assumtions of linear regression:   
1. Linearity between the dependent and the independent columns.
2. Normality of residuals.
3. Heteroscedasticity(constant variance) of residuals.
4. No multicollinearity among the independent columns.
5. No autocorrelation among the residuals.
'''

def check_lineariry(df,target_column):
   

    # separate the X and y from the df

    X = df.drop(columns=target_column)
    y = df[target_column]

    # add constant term to X
    X = sm.add_constant(X)

    # fit the ols model
    ols = sm.OLS(endog=y,exog=X)
    results = ols.fit()

    if results.f_pvalue <= 0.05:
        return 'Reject the null hypothesis, The data is non-linear'
    else:
        return 'Fail to reject the null hypothesis, The data is linear'