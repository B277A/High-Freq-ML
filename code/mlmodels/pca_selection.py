from mlmodels.model import MachineLearningAlgo
import numpy as np
from sklearn.decomposition import PCA as sklearn_pca
import warnings


class PCA_selection(MachineLearningAlgo):

    ## Additional parameters

    def __init__(self, hyperparameters, hyperparameter_grid=None, name="PCA_selection"):
        super().__init__(
            hyperparameters, hyperparameter_grid=hyperparameter_grid, name=name
        )

        if hyperparameter_grid is None:
            # A default value for the hyperparam grid
            self.hyperparameter_grid = {}
            self.hyperparameter_grid["selection_n_components"] = list(range(1, 11))

        self.debug = {}

    def fit(self, Y_ins, X_ins, X_oos, hyperparameters=None):

        if hyperparameters is None:
            warnings.warn("Warning: Using default hyperparameters in fit")
            hyperparameters = self.hyperparameters

        # Ensure number of components is valid
        if hyperparameters["selection_n_components"] > np.shape(X_ins)[1]:
            warnings.warn(
                " ".join(
                    [
                        "Warning: PCA components > number of signals,",
                        " adjusting hyperparameters...",
                    ]
                )
            )
            hyperparameters["selection_n_components"] = np.shape(X_ins)[1]

        sklearn_model = sklearn_pca(
            n_components=hyperparameters["selection_n_components"],
        )

        if np.shape(Y_ins)[1] != 1:
            raise NotImplementedError

        # Model
        sklearn_model_fit = sklearn_model.fit(X_ins)

        # Reduction
        X_ins_sel = sklearn_model_fit.transform(X_ins)
        X_oos_sel = sklearn_model_fit.transform(X_oos)

        # singular values
        singular_val = sklearn_model_fit.singular_values_

        return X_ins_sel, X_oos_sel, singular_val

    def predict(self, X, fit_params):
        pass
