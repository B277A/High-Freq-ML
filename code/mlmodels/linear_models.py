import numpy as np
import logging
import warnings
from mlmodels.model import MachineLearningAlgo
import tensorflow 
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.linear_model import ElasticNet as sklearn_Enet
from sklearn.linear_model import LinearRegression as sklearn_LR
initializers = tensorflow.keras.initializers

# Handle warnings and logs
warnings.simplefilter("error", category=ConvergenceWarning)


class LinearRegression(MachineLearningAlgo):
    def __init__(self, hyperparameters, hyperparameter_grid=None, name="LR"):
        super().__init__(
            hyperparameters, hyperparameter_grid=hyperparameter_grid, name=name
        )

        if hyperparameter_grid is None:
            # A default value for the hyperparam grid
            self.hyperparameter_grid = {}

        if hyperparameters is None:
            self.hyperparameters = {}

    def fit(self, Y_ins, X_ins, X_oos, Y_oos=None, hyperparameters=None, indicator_predict = None):

        sklearn_model = sklearn_LR()

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        if np.shape(X_oos)[1] > np.shape(X_oos)[0]:
            self.logger.debug("Linear Regression does not have full rank: K>>N!")

        # Model
        sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)

        # Predict
        Y_hat = sklearn_model_fit.predict(X_oos)

        # Save params
        coefs, intercept = (
            sklearn_model_fit.coef_[0][:],
            sklearn_model_fit.intercept_[0],
        )
        fit_params = np.insert(coefs, 0, intercept)

        return Y_hat, fit_params


class LASSO(MachineLearningAlgo):

    ## Additional parameters

    def __init__(
        self, hyperparameters, hyperparameter_grid=None, name="Lasso", n_iter=50
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
            self.hyperparameter_grid["lambda"] = np.power(10, np.linspace(-3, 1, 100))
            self.hyperparameter_grid["use_intercept"] = [True]
            self.hyperparameter_grid["seed"] = [666]

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, Y_oos=None, hyperparameters=None, indicator_predict = None):

        if hyperparameters is None:
            self.logger.warning(f"Using default hyperparamters in fit for {self.name}")
            hyperparameters = self.hyperparameters

        sklearn_model = sklearn_Lasso(
            alpha=hyperparameters["lambda"],
            fit_intercept=hyperparameters["use_intercept"],
        )

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Set seed before fit
        np.random.seed(hyperparameters["seed"])

        # Model
        try:
            sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        except ConvergenceWarning:
            # Don't do anything special
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.logger.debug("LASSO Convergence Warning")
                sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        except Exception as e:
            raise (e)

        # Get fit parameters
        fit_params = sklearn_model_fit.coef_

        # Predict
        Y_hat = sklearn_model_fit.predict(X_oos)

        # self.debug['sklearn_model'] = sklearn_model

        return Y_hat, fit_params

    def predict(self, X, fit_params):
        pass

class ENET(MachineLearningAlgo):

    ## Additional parameters

    def __init__(
        self, hyperparameters, hyperparameter_grid=None, name="Enet", n_iter=200
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
            self.hyperparameter_grid["lambda"] = np.power(10, np.linspace(-3, 1, 100))
            self.hyperparameter_grid["use_intercept"] = [True]
            self.hyperparameter_grid["seed"] = [666]
            self.hyperparameter_grid["l1_ratio"] = np.linspace(0.02, 1, 10)

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, Y_oos=None, hyperparameters=None, indicator_predict = None):

        if hyperparameters is None:
            self.logger.warning(f"Using default hyperparamters in fit for {self.name}")
            hyperparameters = self.hyperparameters

        sklearn_model = sklearn_Enet(
            alpha=hyperparameters["lambda"],
            l1_ratio=hyperparameters["l1_ratio"],
            fit_intercept=hyperparameters["use_intercept"],
        )
        

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Set seed before fit
        np.random.seed(hyperparameters["seed"])

        # Model
        try:
            sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        except ConvergenceWarning:
            # Don't do anything special
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.logger.debug("Enet Convergence Warning")
                sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        except Exception as e:
            raise (e)

        # Get fit parameters
        fit_params = sklearn_model_fit.coef_

        # Predict
        Y_hat = sklearn_model_fit.predict(X_oos)

        # self.debug['sklearn_model'] = sklearn_model

        return Y_hat, fit_params

    def predict(self, X, fit_params):
        pass


class LinearTest(MachineLearningAlgo):

    ## Additional parameters

    def __init__(
        self, hyperparameters, hyperparameter_grid=None, name="Lasso", n_iter=50
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
            self.hyperparameter_grid["lambda"] = np.power(10, np.linspace(-3, 1, 2))
            self.hyperparameter_grid["use_intercept"] = [True]
            self.hyperparameter_grid["seed"] = [666]

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, Y_oos=None, hyperparameters=None, indicator_predict = None):

        if hyperparameters is None:
            self.logger.warning(f"Using default hyperparamters in fit for {self.name}")
            hyperparameters = self.hyperparameters

                # Model
        sklearn_model = sklearn_LR()

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Set seed before fit
        np.random.seed(hyperparameters["seed"])

        # Model
        try:
            sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        except ConvergenceWarning:
            # Don't do anything special
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.logger.debug("Convergence Warning")
                sklearn_model_fit = sklearn_model.fit(X_ins, Y_ins)
        except Exception as e:
            raise (e)

        # Get fit parameters
        fit_params = sklearn_model_fit.coef_

        # Predict
        Y_hat = sklearn_model_fit.predict(X_oos)

        # self.debug['sklearn_model'] = sklearn_model

        return Y_hat, fit_params

    def predict(self, X, fit_params):
        pass