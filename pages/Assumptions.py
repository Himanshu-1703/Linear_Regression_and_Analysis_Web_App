import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lr_assumptions import Linearity,Multicollinearity
from lr_assumptions import Normality_Of_Residuals,Autocorrelation
from lr_assumptions import Homoscedasticity

# insert the main page title
st.title('Linear Regression Assumptions App')

# this page is used for linear regression analysis 
st.info('Upload Data in .csv Format')

# upload the .csv file
file = st.file_uploader(label='Upload the df in .csv format only',type=['csv'])

# add a warning for data upload
st.warning('''
           1. The file size should not be greater than 10 MB.
           2. The data should not contain any missing values.
           3. The input columns should be scaled before hand to give good results.
           4. Feature selection and Feature preprocessing to be done before uploading the file.
           ''')

st.sidebar.title('Menu')

if file:
    df = pd.read_csv(file)

    # output column selection from the data
    target_col = st.sidebar.selectbox(label='Output column in DataFrame',options=df.columns,
                        index=len(df.columns)-1,help='Choose a Continuous column')

    # make X and y in the data
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    
    # make instances of all assumptions
    
    lin = Linearity(X,y)
    multi = Multicollinearity(X,y)
    norm = Normality_Of_Residuals(X,y)
    auto = Autocorrelation(X,y)
    homo = Homoscedasticity(X,y)
    
    options_dict = {'Linearity of features':lin,
                    'Multicollinearity':multi,
                    'Normality of Residuals':norm,
                    'Autocorrelation of Residuals':auto,
                    'Homoscedasticity':homo}
    
    # select the appropriate assumption 
    assumption = st.sidebar.selectbox(label='Select the Regression Assumption',
                         options=list(options_dict.keys()))
    
    if assumption == 'Linearity of features':
        pass
    
    elif assumption == 'Multicollinearity':
        pass
    
    elif assumption == 'Normality of Residuals':
        pass
    
    elif assumption == 'Autocorrelation of Residuals':
        pass
    
    elif assumption == 'Homoscedasticity':
        pass
    
    