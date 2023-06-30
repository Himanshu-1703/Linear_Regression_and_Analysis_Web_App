from gd_regressor import BGD_Regressor
from gd_regressor import SGD_Regressor
from gd_regressor import MBGD_Regressor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class LR:
    '''
    Linear Regression class for prediction and inference purposes using different methods.

    Parameters
        method : str, default='OLS'
        The method to use for fitting the model. Can be 'OLS' or 'GD'.

        purpose : str, default='prediction'
        The purpose of the model. Can be 'prediction' or 'inference'.

        gd_regressor_type : str, default=None
        The type of gradient descent regressor to use if method is 'GD'. Can be 'batch_gd', 'stochastic_gd', or 'mini_batch_gd'.

        epochs : int, default=None
        The number of epochs to use for training the model if method is 'GD'.

        learning_rate : float, default=None
        The learning rate to use for training the model if method is 'GD'.

        batch_size : int, default=None
        The batch size to use for training the model if method is 'GD' and gd_regressor_type is 'mini_batch_gd'.

    Methods:
        fit(X, y):
            Fits a linear regression model to the given input data and target values.

            Args:
                X (array-like): The input data of shape (n_samples, n_features).
                y (array-like): The target values of shape (n_samples,).

        predict(X):
            Predicts the target values for the given input data.

            Args:
                X (array-like): The input data of shape (n_samples, n_features).

            Returns:
                y_pred (array-like): The predicted target values of shape (n_samples,).

        score(y_true, y_pred):
            Calculates the coefficient of determination (R^2) for the predicted target values.

            Args:
                y_true (array-like): The true target values of shape (n_samples,).
                y_pred (array-like): The predicted target values of shape (n_samples,).

            Returns:
                score (float): The coefficient of determination (R^2) score. Returns None if method is 'OLS' and purpose is 'inference'.
    '''




    def __init__(self,method='OLS',purpose='prediction',gd_regressor_type=None,
                 epochs=None,learning_rate=None,batch_size=None):
        self.method = method
        self.purpose = purpose
        self.gd_regressor_type = gd_regressor_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lin_obj = None


    def fit(self,X,y):
        if (self.method == 'OLS') and (self.purpose == 'prediction'):
            # fit a linear regression
            self.lin_obj = LinearRegression()
            self.lin_obj.fit(X,y)

        elif (self.method == 'OLS') and (self.purpose == 'inference'):
            # fit statsmodels OLS
            X = sm.add_constant(X)
            self.lin_obj = sm.OLS(y,X)
            results = self.lin_obj.fit()

            # generate summary
            print(results.summary())

        elif (self.method == 'GD') and (self.purpose == 'prediction'):
            if self.gd_regressor_type == 'batch_gd':
                # fit a batch gradient descent regressor
                self.lin_obj = BGD_Regressor(epochs=self.epochs,learning_rate=self.learning_rate)
                self.lin_obj.fit(X,y)

            elif self.gd_regressor_type == 'stochastic_gd':
                # fit a SGD regressor
                self.lin_obj = SGD_Regressor(epochs=self.epochs,learning_rate=self.learning_rate)
                self.lin_obj.fit(X,y)

            elif self.gd_regressor_type == 'mini_batch_gd':
                self.lin_obj = MBGD_Regressor(epochs=self.epochs,learning_rate=self.learning_rate,batch_size=self.batch_size)
                self.lin_obj.fit(X,y)


    def predict(self,X):
        if (self.method == 'OLS') and (self.purpose == 'inference'):
            y_pred = self.lin_obj.predict(endog=X)
            return y_pred

        else:
            y_pred = self.lin_obj.predict(X)
            return y_pred

    
    def score(self,y_true,y_pred):
        if (self.method == 'OLS') and (self.purpose == 'inference'):
            return None

        elif (self.method == 'OLS') and (self.purpose == 'prediction'):
            score = r2_score(y_true,y_pred)
            return score
        else:
            score = self.lin_obj.score(y_true,y_pred)
            return score
        

   