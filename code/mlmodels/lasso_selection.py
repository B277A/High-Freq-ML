import numpy as np
import warnings
import logging
from mlmodels.model import MachineLearningAlgo
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso as sklearn_Lasso

# Handle warnings and logs
warnings.simplefilter("error", category=ConvergenceWarning)


class LASSO_selection(MachineLearningAlgo):

    ## Additional parameters

    def __init__(
        self, hyperparameters, hyperparameter_grid=None, name="Lasso_selection"
    ):
        super().__init__(
            hyperparameters, hyperparameter_grid=hyperparameter_grid, name=name
        )

        if hyperparameter_grid is None:
            # A default value for the hyperparam grid
            self.hyperparameter_grid = {}
            self.hyperparameter_grid["selection_lambda"] = np.power(
                10, np.linspace(-3, 1, 40)
            )
            self.hyperparameter_grid["selection_use_intercept"] = [True]
            self.hyperparameter_grid["selection_seed"] = [666]

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, Y_oos=None, hyperparameters=None, indicator_predict = None):

        if hyperparameters is None:
            self.logger.warning(f"Using default hyperparamters in fit for {self.name}")
            hyperparameters = self.hyperparameters

        sklearn_model = sklearn_Lasso(
            alpha=hyperparameters["selection_lambda"],
            fit_intercept=hyperparameters["selection_use_intercept"],
        )

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Set seed before fit
        np.random.seed(hyperparameters["selection_seed"])

        # Selection model
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

        # selection
        coef_indicater = abs(sklearn_model_fit.coef_) > 0
        X_ins_sel = X_ins.iloc[:, coef_indicater]
        X_oos_sel = X_oos.iloc[:, coef_indicater]

        return X_ins_sel, X_oos_sel, coef_indicater

    def predict(self, X, fit_params):
        pass
