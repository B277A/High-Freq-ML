import numpy as np
import logging
import warnings
from sklearn.ensemble import RandomForestRegressor as sklearn_RF
from mlmodels.model import MachineLearningAlgo


class RandomForest(MachineLearningAlgo):

    ## Additional parameters

    def __init__(
        self, hyperparameters, hyperparameter_grid=None, name="RF", n_iter=6
    ):
        super().__init__(
            hyperparameters,
            hyperparameter_grid=hyperparameter_grid,
            name=name,
            n_iter=n_iter,
        )

        if hyperparameter_grid is None:
            # A default value for the hyperparam grid
            self.hyperparameter_grid = {}
            self.hyperparameter_grid["n_tree"] = [500, 800,  1000]
            self.hyperparameter_grid["seed"] = [0, 666]

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, hyperparameters=None):

        if hyperparameters is None:
            self.logger.warning("Warning: Using default hyperparameters in fit")
            hyperparameters = self.hyperparameters

        self.logger.debug("Warning: Mathias the hyperparameteres are shit! Fix it!")
        sklearn_model = sklearn_RF(
            n_estimators=hyperparameters["n_tree"],
            random_state=hyperparameters["seed"],
        )

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Set seed before fit
        np.random.seed(hyperparameters["seed"])

        # Model
        sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins.values.flatten())

        # Predict
        Y_hat = sklearn_model_fit.predict(X_oos)

        # self.debug['sklearn_model'] = sklearn_model
        fit_params = {}

        return Y_hat, fit_params
