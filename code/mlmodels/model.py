import numpy as np
import logging

# ML Algo classes
class MachineLearningAlgo:
    r"""
    Generic Machine Learning Algorithm; functions as a superclass for
    other algorithms.

    Parameters
    ----------
    hyperparameters : dict
         Contains...
    hyperparameter_grid : dict
         Defines a range of parameter values to test when doing validation

    Notes
    -----
    ...
    """

    def __init__(
        self, hyperparameters, hyperparameter_grid=None, name=None, n_iter=None
    ):

        self.hyperparameters = hyperparameters

        if hyperparameter_grid is None:
            self.hyperparameter_grid = {}
        else:
            self.hyperparameter_grid = hyperparameter_grid

        self.name = name
        self.n_iter = n_iter

        # Logger set to main, simpler than specifying specific name
        self.logger = logging.getLogger("__main__")

    def fit(self, Y, X, Z, hyperparameters=None):
        raise NotImplementedError


class PostSelectionModel(MachineLearningAlgo):
    def __init__(self, model_selection, model_forecast, name=None, n_iter=None):

        if not n_iter:
            n_iter = model_forecast.n_iter

        if not name:
            name = (
                f"PostSelectionModel({model_selection.name} -> {model_forecast.name})"
            )

        super().__init__(None, None, name=name, n_iter=n_iter)

        # Hold onto model pointers
        self.model_selection = model_selection
        self.model_forecast = model_forecast

        # Set up hyperparameters
        self.hyperparameter_grid = {
            **model_selection.hyperparameter_grid,
            **model_forecast.hyperparameter_grid,
        }

        # Make sure inputs are okay
        self.check_inputs()

    def check_inputs(self):
        # Check inputs to this model - could expand on this

        # Remember we need to differentiate
        # selection hyperparams from forecast hyperparams
        for key in self.model_selection.hyperparameter_grid:
            if "selection_" not in key:
                print(
                    "Hyperparameter grid keys:"
                    + f"{self.model_selection.hyperparameter_grid.keys()}"
                )
                exception_str = (
                    f'The suffix "selection_" is not a part '
                    + f"of the following hyperparameter key: {key}"
                )

                raise Exception(exception_str)

        pass

    def fit(self, Y_ins, X_ins, X_oos, hyperparameters=None):

        if hyperparameters is None:
            self.logger.warning(f"Using default hyperparamters in fit for {self.name}")
            hyperparameters = self.hyperparameters

        # First step is to use selection to cut down X
        X_ins_sel, X_oos_sel, selection_fit_params = self.model_selection.fit(
            Y_ins, X_ins, X_oos, hyperparameters=hyperparameters
        )

        # If dimensionality reduction totally drops all columns
        if np.shape(X_ins_sel)[1] == 0:
            # Define new columns as just a column of ones
            X_ins_sel = X_ins.iloc[:, [0]] * 0 + 1
            X_oos_sel = X_oos.iloc[:, [0]] * 0 + 1

        # Second step is to use selected X to forecast
        Y_hat, forecast_fit_params = self.model_forecast.fit(
            Y_ins, X_ins_sel, X_oos_sel, hyperparameters=hyperparameters
        )

        fit_params = {
            "selection": selection_fit_params,
            "forecast": forecast_fit_params,
        }

        return Y_hat, fit_params
