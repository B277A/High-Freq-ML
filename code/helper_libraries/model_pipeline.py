import pandas as pd
import numpy as np
import warnings
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics

# Model Trainer class
class ModelTrainer:
    r"""
    Use training data to validate the model.

    Parameters
    ----------
    model_list : list
        List of machine learning algo objects
    Y_ins : dataframe
        In-sample data (dependent)
    X_ins : dataframe
        In-sample data (independent)
    seed : int
        Random seed for the hyperparameter draws
    check_data : boolean
        If true, checks the input data for errors

    Notes
    -----
    ...
    """

    def __init__(self, model_list, Y_ins, X_ins, seed=0, check_data=False):
        self.model_list = model_list
        self.Y_ins = Y_ins
        self.X_ins = X_ins
        self.seed = seed

        # Only check data if requested
        if check_data:
            self.check_data()

    def check_data(self):

        # Finite values?
        sklearn.utils.assert_all_finite(self.X_ins)
        sklearn.utils.assert_all_finite(self.Y_ins)

        # Could put other tests here
        # ...

    def split_data(self, frac):
        # Simple split into train and validate
        # Fraction refers to size of validation
        # relative to entire length of Y

        # Define in-sample and oos data
        n_obs = np.shape(self.X_ins)[0] + 1
        split_idx = int(np.ceil(n_obs * frac))
        X_train = self.X_ins.iloc[:split_idx, :]
        Y_train = self.Y_ins.iloc[:split_idx, :]
        X_valid = self.X_ins.iloc[split_idx:, :]
        Y_valid = self.Y_ins.iloc[split_idx:, :]

        return Y_train, X_train, Y_valid, X_valid

    def scale_train_valid_data(self, Y_train, X_train, Y_valid, X_valid):
        # Rescales training and validation data
        # Saves scaler objects for access and untransform later

        # Create scaler objects
        scaler_X_train = sklearn.preprocessing.StandardScaler()
        scaler_X_train.fit(X_train)
        scaler_Y_train = sklearn.preprocessing.StandardScaler()
        scaler_Y_train.fit(Y_train)

        # Rescale data
        X_train_scl = self.scale_dataframe(X_train, scaler_X_train)
        X_valid_scl = self.scale_dataframe(X_valid, scaler_X_train)
        Y_train_scl = self.scale_dataframe(Y_train, scaler_Y_train)
        Y_valid_scl = self.scale_dataframe(Y_valid, scaler_Y_train)

        # Save scalers for later
        self.scaler_X_train = scaler_X_train
        self.scaler_Y_train = scaler_Y_train

        return Y_train_scl, X_train_scl, Y_valid_scl, X_valid_scl

    def validation(self, frac=0.8, n_iter=10):
        # frac is size of train data
        # n_iter is number of hyperparam draws

        # Split up data
        Y_train, X_train, Y_valid, X_valid = self.split_data(frac)

        # Standardize data
        Y_train_scl, X_train_scl, Y_valid_scl, X_valid_scl = self.scale_train_valid_data(
            Y_train, X_train, Y_valid, X_valid
        )
        self.Y_train_scl = Y_train_scl
        self.X_train_scl = X_train_scl
        self.Y_valid_scl = Y_valid_scl
        self.X_valid_scl = X_valid_scl

        # Output list of optimal hyperparameters
        model_hyperparameters_opt = []

        # Validate models
        for model in self.model_list:

            # List of possible hyperparameters and their values
            # for the model
            hyperparameter_grid = model.hyperparameter_grid

            # Objective function values
            error_list = []

            # Grab random hyperparameters
            for hyperparam_draw in sklearn.model_selection.ParameterSampler(
                hyperparameter_grid, n_iter=n_iter, random_state=self.seed
            ):

                # Fit the model using these hyperparameters
                Y_hat_scl, _ = model.fit(
                    Y_train_scl,
                    X_train_scl,
                    X_valid_scl,
                    hyperparameters=hyperparam_draw,
                )

                # Rescale output
                Y_hat = self.scaler_Y_train.inverse_transform(Y_hat_scl)

                error_list.append(
                    [
                        hyperparam_draw,
                        self.objective_function(Y_valid, Y_hat, function="mse"),
                    ]
                )

            hyperparameters_opt = (
                pd.DataFrame(error_list, columns=["hyperparameters", "error"])
                .sort_values(by="error")
                .iloc[0]["hyperparameters"]
            )
            model_hyperparameters_opt.append(hyperparameters_opt)

            # DEBUG
            self.error_list = pd.DataFrame(
                error_list, columns=["hyperparameters", "error"]
            ).sort_values(by="error")

        self.model_hyperparameters_opt = model_hyperparameters_opt

    def crossvalidation(self, K=5):
        pass

    @staticmethod
    def scale_dataframe(df, scaler, inverse=False):
        if not inverse:
            return pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
        else:
            return pd.DataFrame(scaler.inverse_transform(df), index=df.index, columns=df.columns)

    @staticmethod
    def objective_function(Y_true, Y_est, function="mse"):
        if function == "mse":
            return sklearn.metrics.mean_squared_error(Y_true, Y_est, multioutput='uniform_average')


# Model Tester class
class ModelTester:
    r"""
    Use testing data to forecast the model.

    Parameters
    ----------
    modeltrainer : ModelTrainer
        Object with already trained models

    Notes
    -----
    ...
    """

    def __init__(self, modeltrainer):
        self.modeltrainer = modeltrainer

    def forecast(self, Y_test, X_test):

        # Standardize the testing X data
        X_test_scl = self.modeltrainer.scale_dataframe(
            X_test, self.modeltrainer.scaler_X_train
        )
        self.X_test_scl = X_test_scl

        # Standardized training data from model trainer
        Y_train_scl = self.modeltrainer.Y_train_scl
        X_train_scl = self.modeltrainer.X_train_scl

        # Output list; forecasts using each model
        model_forecasts_list = []

        # For each model, make a forecast
        for i in range(len(self.modeltrainer.model_list)):

            model = self.modeltrainer.model_list[i]
            hyperparameters = self.modeltrainer.model_hyperparameters_opt[i]

            Y_hat_scl, _ = model.fit(
                Y_train_scl, X_train_scl, X_test_scl, hyperparameters=hyperparameters
            )

            Y_hat = self.modeltrainer.scale_dataframe(
                pd.DataFrame(Y_hat_scl, index=Y_test.index, columns=Y_test.columns),
                self.modeltrainer.scaler_Y_train,
                inverse=True,
            )

            model_forecasts_list.append(Y_hat)

        return model_forecasts_list