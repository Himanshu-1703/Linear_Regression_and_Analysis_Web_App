import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regression import LR


# insert the main page title
st.title('Linear Regression App')

# this page is used for linear regression analysis 
st.info('Upload Data in .csv Format')

# upload the .csv file
file = st.file_uploader(label='Upload the df in .csv format only',type=['csv'])

df = pd.read_csv(file)

# add a warning for data upload
st.warning('''
           1. The file size should not be greater than 10 MB.
           2. The data should not contain any missing values.
           3. The input columns should be scaled before hand to give good results.
           4. Feature selection and Feature preprocessing to be done before uploading the file.
           ''')

st.sidebar.title('Menu')

# output column selection from the data
target_col = st.sidebar.selectbox(label='Output column in DataFrame',options=df.columns,
                     index=len(df.columns)-1,help='Choose a Continuous column')

# make X and y in the data
X = df.drop(columns=target_col)
y = df[target_col]

# train test split the data
st.sidebar.subheader('Choose the Test Size ratio for the split')
test_ratio = st.sidebar.slider(label='Test Size ratio',min_value=10,
                  max_value=90,value=20,step=10,
                  help='The train size ratio will be (100-Test Ratio)')

# introduce randomness everytime we make a split
rand_no = np.random.randint(1,1000) 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_ratio,random_state=rand_no)


# choose the algorithm for predictions
st.sidebar.header('Choose the Parameters for the Algorithm')


def method_label(option):
    if option == 'OLS':
        return 'OLS'
    else:
        return 'Gradient Descent'
    

# for the method of algorithm
method_name = st.sidebar.selectbox('Select the appropriate method for prediction',options=['OLS','GD'],
                     format_func=method_label,index=0)

flag_method = False

if method_name == 'GD':
    flag_method = True

# for the purpose of prediction
prediction_type = 'prediction'
prediction_type = st.sidebar.selectbox('Select the Type of model',options=['prediction','inference'],index=0,
                                       format_func=lambda x:x.capitalize(),disabled=flag_method)

def gd_label(option):
    if option in ['batch_gd', 'stochastic_gd']:
        split_list = option.split('_')
        name = split_list[0].capitalize()
        end = split_list[1].upper()
        return name + ' ' + end
    
    elif option == 'mini_batch_gd':
        split_list = option.split('_')
        name = split_list[0].capitalize()
        middle = split_list[1].capitalize()
        end = split_list[2].upper()
        return name + ' ' + middle + ' ' + end

# make options if the method slected is GD

if (method_name == 'GD') and (prediction_type=='prediction'):
    gd_type = st.sidebar.selectbox('Select the Type of Gradient Descent method',
                                   options=['batch_gd', 'stochastic_gd', 'mini_batch_gd'],
                                   index=0,format_func=gd_label)
    
    epochs = st.sidebar.number_input('Enter the Number of Epochs',min_value=10,max_value=10000,
                                     value=100,step=1)
    
    if type(epochs) != int:
        st.error('Floating point Epochs value not supported')
        
    learning_rate = st.sidebar.number_input('Enter the Learning Rate',min_value=0.01,max_value=10.0,
                                     value=0.1,step=0.01)

    if gd_type == 'mini_batch_gd':
        batch_size = st.sidebar.number_input('Enter the Batch Size',min_value=32,max_value=2096,
                                     value=32,step=32)
        
        
    
# define the linear regression object
#linear_reg = LR()
