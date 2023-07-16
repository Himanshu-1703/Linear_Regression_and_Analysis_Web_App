import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regression import LR
import plotly.express as px
import plotly.graph_objects as go


# insert the main page title
st.title('Linear Regression App')

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

    # train test split the data
    st.sidebar.subheader('Choose the Test Size ratio for the split')
    test_ratio = st.sidebar.slider(label='Test Size ratio',min_value=0.1,
                    max_value=0.9,value=0.2,step=0.1,
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

        batch_size  = 32
        
        if gd_type == 'mini_batch_gd':
            batch_size = st.sidebar.number_input('Enter the Batch Size',min_value=32,max_value=2096,
                                        value=32,step=32)
            
    # plot the graph for the regression line
    
    if prediction_type == 'prediction':
            
            # use the radio button for plotting the regression line
            radio = st.sidebar.radio(label='Plot the regression plot in 2D/3D',
                                    options=['2D','3D'],index=1)
            
       
    btn = st.sidebar.button("Apply")
    
    # toggle everything if the button is pressed
    if btn:
        # define the linear regression object
        if prediction_type == 'prediction':
            if method_name == 'OLS':
                linear_reg = LR(method=method_name,purpose=prediction_type)
                
            elif method_name == 'GD':
                linear_reg = LR(method=method_name,purpose=prediction_type,
                                gd_regressor_type=gd_type,epochs=epochs,
                                learning_rate=learning_rate,batch_size=batch_size)
            
            
            # fit the model
            linear_reg.fit(X_train,y_train)
        
            # make predictions on the test data
            y_pred = linear_reg.predict(X_test)
         
            # get a score for the regression model
            score = linear_reg.score(y_test,y_pred)
            
            # print the score for the model
            st.subheader(f"The score of the Regression Model is {np.round(score,2)}")
            
             # toggle it only when OLS and Gradient Descent are used
            if radio == "2D":
                    
                # perform PCA on the data
                from sklearn.decomposition import PCA

                pca = PCA(n_components=1)
                
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                
                # fit the model
                linear_reg.fit(X_train_pca.reshape(-1,1),y_train)
                
                # predict on synthetic data
                X_test_temp = np.linspace(X_train_pca.min(),X_train_pca.max(),100)
                
                y_pred_plot = linear_reg.predict(X_test_temp.reshape(-1,1))
                
                # plot the graph
                fig, ax = plt.subplots()
                ax.scatter(X_train_pca,y_train)
                
                # plot the regression line
                ax.plot(X_test_temp,y_pred_plot,color='red')
                
                st.pyplot(fig)
                    
            elif radio == "3D":
                # perform PCA on the data
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                
                # fit the model
                linear_reg.fit(X_train_pca,y_train)
                
                # make the mehgrid
                x = np.linspace(X_train_pca[:,0].min(),X_train_pca[:,0].max(),500)
                y = np.linspace(X_train_pca[:,1].min(),X_train_pca[:,1].max(),500)
                
                XX,YY = np.meshgrid(x,y)
                
                # make the array
                arr = np.array([XX.ravel(),YY.ravel()]).T
                
                # make predictions on the arr
                z = linear_reg.predict(arr)
                z = z.reshape(XX.shape)
                
                # plot the contour plot and scatter plot
                fig = px.scatter_3d(x=X_train_pca[:,0],y=X_train_pca[:,1],
                                    z=y_train)
                
                fig.add_trace(go.Surface(x=x,y=y,z=z))
                
                st.plotly_chart(fig)
                
        elif prediction_type =='inference':
            linear_reg = LR(method=method_name,purpose=prediction_type)
            
            # fit the OLS inference model
            summary = linear_reg.fit(X_train,y_train)
            st.subheader('Regression Analysis Summary')
            st.write(summary)

    
        
    