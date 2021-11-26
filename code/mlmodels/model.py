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

    def __init__(self, hyperparameters, hyperparameter_grid=None, name=None):

        self.hyperparameters = hyperparameters

        if hyperparameter_grid is None:
            self.hyperparameter_grid = {}
        else:
            self.hyperparameter_grid = hyperparameter_grid

        self.name = name

    def fit(self, Y, X):
        raise NotImplementedError