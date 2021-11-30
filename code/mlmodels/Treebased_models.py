from mlmodels.model import MachineLearningAlgo
import numpy as np
from sklearn.ensemble import RandomForestRegressor as sklearn_RF
from sklearn.linear_model import LinearRegression as sklearn_LR
import warnings

    
class RandomForrest(MachineLearningAlgo):

    ## Additional parameters

    def __init__(self, hyperparameters, hyperparameter_grid=None, name="Lasso"):
        super().__init__(hyperparameters, hyperparameter_grid)

        if hyperparameter_grid is None:
            # A default value for the hyperparam grid
            self.hyperparameter_grid = {}
            self.hyperparameter_grid["use_intercept"] = [False, True]
            self.hyperparameter_grid["seed"] = [0, 666]

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, hyperparameters=None):

        if hyperparameters is None:
            warnings.warn("Warning: Using default hyperparameters in fit")
            hyperparameters = self.hyperparameters

        sklearn_model = sklearn_RF(max_depth=2, random_state=0)

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Set seed before fit
        np.random.seed(hyperparameters["seed"])
            
        # Model
        sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        fit_params = None
        
        # Predict
        Y_hat = sklearn_model_fit.predict(X_oos)

        # self.debug['sklearn_model'] = sklearn_model

        return Y_hat, fit_params