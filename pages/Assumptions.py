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

# set the title for the sidebar
st.sidebar.title('Menu')

# check if the file is uploaded
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
    
    # make a dictionary of all the object instances
    options_dict = {'Linearity of features':lin,
                    'Multicollinearity':multi,
                    'Normality of Residuals':norm,
                    'Autocorrelation of Residuals':auto,
                    'Homoscedasticity':homo}
    
    # select the appropriate assumption 
    assumption = st.sidebar.selectbox(label='Select the Regression Assumption',
                         options=list(options_dict.keys()))
    
    # header for the assumptions output
    st.subheader('Output of the test')
    
    if assumption == 'Linearity of features':
        
        # dictionary of the methods in linearity class
        lin_dict = {'Plot Residual Plot':lin.plot_residplot, 
                    'Polynomial Transformation':lin.fit_polynomial, 
                    'Plot Scatterplots':lin.plot_linearity}
        
        # choose the test
        option = st.sidebar.selectbox(label='Choose the test',
                             options=list(lin_dict.keys()))
        
        # plot the graphs
        if option in list(lin_dict.keys())[::2]:
            btn = st.sidebar.button(label='Plot Graph')
            if btn:
                fig = lin_dict.get(option)()
                st.pyplot(fig)
        
        # use polynomial tranformation on the data to compare with linear regression
        elif option in list(lin_dict.keys())[1]:
            degree_slider = st.sidebar.slider(label='Select the degree of polynomial terms',
                              min_value=2,max_value=10,value=2,
                              step=1)
            btn = st.sidebar.button(label='Apply Polynomial')
            
            # activate on press of button
            if btn:
                res1,res2 = lin_dict.get(option)(degree=degree_slider)
                
                st.write(f'The R2 score without Polynomial Transformation is {np.round(res1,2)}')
                st.write(f'The R2 score with Polynomial Transformation is {np.round(res2,2)}')
    
    
    elif assumption == 'Multicollinearity':
        
        # create the dictionary for multicollinearity methods
        multi_dict = {'Plot Correlation Matrix':multi.plot_corr_matrix, 
                      'Perform VIF Test':multi.calculate_vif}

        option = st.sidebar.selectbox(label='Choose the test',
                             options=list(multi_dict.keys()))
        
        # plot the correlation matrix
        if option == 'Plot Correlation Matrix':
            btn = st.sidebar.button(label='Plot Graph')
            if btn:
                fig = multi_dict.get(option)()
                st.pyplot(fig)
        
        # get the VIF score dataframe        
        elif option == 'Perform VIF Test':
            btn = st.sidebar.button(label='Do VIF Test')
            if btn:
                vif_df = multi_dict.get(option)()
                st.write('The results of the VIF Test are:')
                st.dataframe(vif_df)
    
    
    elif assumption == 'Normality of Residuals':
        
        # create dictionary for methods for normality instance
        norm_dict = {'Plot Histogram':norm.plot_graph, 
                     'Plot QQ Plot':norm.plot_qq,
                     'Shapiro test':norm.perf_shapiro, 
                     'Omnibus Test':norm.perf_omnibus, 
                     'Jarque Bera Test':norm.perf_jarque_bera}

        # choose the test in sidebar
        option = st.sidebar.selectbox(label='Choose the test',
                             options=list(norm_dict.keys()))
        
        # plot graphs
        if option in list(norm_dict.keys())[0:2]:
            btn = st.sidebar.button(label='Plot Graph')
            if btn:
                fig,skew = norm_dict.get(option)()
                st.pyplot(fig)
                st.write(f'The skewness of the Distribution is {skew}')
        
        # perform hypothesis test for normality
        elif option in list(norm_dict.keys())[2:]:
            btn = st.sidebar.button(label='Perform Hypothesis test')
            if btn:
                res,p_val = norm_dict.get(option)()
                st.write('The result of the Hypothesis Test is:')
                st.write(f'{res} and the p value is {p_val}')
                
    
    
    elif assumption == 'Autocorrelation of Residuals':
        option = st.sidebar.selectbox(label='Choose the test',
                             options=['Autocorrelation of Residuals'])
        
        # plot the autocorrelation graph
        btn = st.sidebar.button(label='Plot Graph')
        if btn:
            fig = auto.plot_autocorrelation()
            st.pyplot(fig)
    
    elif assumption == 'Homoscedasticity':
        option = st.sidebar.selectbox(label='Choose the test',
                             options=['Plot Residual Plot'])
        
        # plot residual plot
        btn = st.sidebar.button(label='Plot Graph')
        if btn:
            fig = homo.plot_residplot()
            st.pyplot(fig)
                
    