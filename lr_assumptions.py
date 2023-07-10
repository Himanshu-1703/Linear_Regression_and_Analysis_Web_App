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

def check_linearity(df:pd.DataFrame,target_column:pd.Series):
   # add constant term to X
    X = sm.add_constant(df)

    # fit the ols model
    ols = sm.OLS(endog=target_column,exog=X)
    results = ols.fit()

    if results.f_pvalue <= 0.05:
        return 'Reject the null hypothesis, The data shows a linear relationship with the target'
    else:
        return 'Fail to reject the null hypothesis, The data does not show any linear relationship with the target'
    
    
def plot_linearity(X:pd.DataFrame,target_column:pd.Series):
    # select the numerical columns
    num_data = X.select_dtypes(include=np.number)
    
    # number of columns in input data
    num_cols = len(X.columns)
    
    # fig object and size
    fig = plt.figure(figsize=(13,(num_cols//2)*8))

        
    for ind,col in enumerate(num_data.columns):
        # plot the subplot
            plt.subplot((num_cols//2)+1,2,ind+1)
            
            # plot the scatter plot
            sns.scatterplot(data=X,x=col,y=target_column)
            plt.tight_layout()
            
    return fig
        
        