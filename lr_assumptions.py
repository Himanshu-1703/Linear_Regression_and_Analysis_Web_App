import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro,probplot
from statsmodels.stats.outliers_influence import variance_inflation_factor




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


class Linearity:
    def __init__(self,df:pd.DataFrame,target_column:pd.Series) -> None:
        self.df = df
        self.target = target_column


    def check_linearity(self):
        # add constant term to X
        X = sm.add_constant(self.df)

        # fit the ols model
        ols = sm.OLS(endog=self.target,exog=X)
        results = ols.fit()

        if results.f_pvalue <= 0.05:
            return 'Reject the null hypothesis, The data shows a linear relationship with the target'
        else:
            return 'Fail to reject the null hypothesis, The data does not show any linear relationship with the target'
        
        
    def plot_linearity(self):
        # select the numerical columns
        num_data = self.df.select_dtypes(include=np.number)
        
        # number of columns in input data
        num_cols = len(self.df.columns)
        
        # fig object and size
        fig = plt.figure(figsize=(13,(num_cols//2)*8))

            
        for ind,col in enumerate(num_data.columns):
            # plot the subplot
                plt.subplot((num_cols//2)+1,2,ind+1)
                
                # plot the scatter plot
                sns.scatterplot(data=self.df,x=col,y=self.target)
                plt.tight_layout()
                
        return fig
        





class Normality_Of_Residuals:
    
    def __init__(self,df:pd.DataFrame,target_column:pd.Series) -> None:
        self.df = df
        self.target = target_column    
    
    def __calculate_residuals(self):
       lin_reg = LinearRegression()
       
       # fit on the data
       lin_reg.fit(self.df,self.target)
       
       # make predictions
       y_pred = lin_reg.predict(self.df)
       
       # calaculate the residuals(errors)
       resid = self.target - y_pred 
       return resid
       
       
    def plot_graph(self):
        residuals = self.__calculate_residuals()
        
        # plot the histogram and kde plot for the residuals

        fig = sns.histplot(residuals,kde=True)
        return fig
    
    def perf_test(self):
        residuals = self.__calculate_residuals()
        
        # perform the hypothesis test
        stats,p_value = shapiro(residuals.values)
        
        if p_value > 0.05:
            print('Fail to reject the Null Hypothesis. The residuals are normally distributed')
            
        else:
            print('Reject the Null Hypothesis. The residuals are not norally distributed')
            
    
    def plot_qq(self):
        residuals = self.__calculate_residuals()
        
        # set the figure parameters
        fig = plt.figure(figsize=(10,6))
        probplot(residuals.values,plot=fig)
        plt.title('QQ plot of residuals')
        
        return fig
        
        

class Homoscedasticity:
    def __init__(self,df:pd.DataFrame,target_column:pd.Series) -> None:
        self.df = df
        self.target = target_column    
    
    def __calculate_residuals(self):
       lin_reg = LinearRegression()
       
       # fit on the data
       lin_reg.fit(self.df,self.target)
       
       # make predictions
       y_pred = lin_reg.predict(self.df)
       
       # calaculate the residuals(errors)
       resid = self.target - y_pred 
       return resid
       
    def plot_residplot(self):
        residuals = self.__calculate_residuals()
        
        # plot the residual to create a residual plot
        fig = plt.figure(figsize=(12,5))
        plt.scatter(self.target.values,residuals,color='green')
        plt.axhline(y=0,linestyle='--')
        plt.xlabel('y_pred')
        plt.ylabel('residuals')
        plt.title('Residual plot')
        
        return fig
    
    
    
class Multicollinearity:
    def __init__(self,df:pd.DataFrame,target_column:pd.Series) -> None:
        self.df = df
        self.target = target_column    
        
        
    def plot_corr_matrix(self):
        fig = plt.figure(figsize=(10,10))
        
        # plot the heatmap
        sns.heatmap(self.df.corr(),annot=True,cmap='RdBu')
        
        return fig
    
    
    def calculate_vif(self) -> pd.DataFrame:
        vif_scores = []
        
        for col_no in range(len(self.df.columns)):
            val = variance_inflation_factor(self.df,col_no)
            vif_scores.append(val)
            
        vif_df = pd.DataFrame({'VIF':vif_scores},index=self.df.columns)
        
        return vif_df
    
    
    
class Autocorrelation:
    def __init__(self,df:pd.DataFrame,target_column:pd.Series) -> None:
        self.df = df
        self.target = target_column    
     
    def __calculate_residuals(self):
       lin_reg = LinearRegression()
       
       # fit on the data
       lin_reg.fit(self.df,self.target)
       
       # make predictions
       y_pred = lin_reg.predict(self.df)
       
       # calaculate the residuals(errors)
       resid = self.target - y_pred 
       return resid    
    
    def plot_autocorrelation(self):
        residuals = self.__calculate_residuals()
        
        # plot the autocorrelation graph:
        fig = plt.figure(figsize=(12,6))
        
        plt.plot(residuals)
        plt.title('Autocorrelation of Residuals')
        
        return fig