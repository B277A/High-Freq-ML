# Add other paths
import os, sys
sys.path.append(os.path.abspath(os.path.join('../')))

# Should be default
import pandas as pd
import numpy as np
import logging
import logging.config
import warnings
import datetime as dt
import multiprocessing
from multiprocessing import Pool

# Stuff you need to have on the server
from tqdm.auto import tqdm
from sklearn import linear_model
import seaborn as sns
import sklearn.preprocessing
import sklearn.utils
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Our libraries
from mlmodels.model import *
from mlmodels.linear_models import *
from mlmodels.treebased_models import *
from mlmodels.pca_selection import *
from mlmodels.lasso_selection import *
from helper_libraries.model_pipeline import *
from helper_libraries.forecasting_tools import *
from helper_libraries.preprocessing_tools import *

## Params

# Directory
main_directory = '../..'


# Start logging
log_filename = f"{main_directory}/code/logs/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_test_postselection.log"
logging.config.fileConfig("../_config/logging.conf", defaults={'log_filename': log_filename})


### Data

# Set up some sample data for the tests
sample_data_df = pd.read_parquet('../../data/proc/_temp/1996_all.parquet')

# Some data that comes with seaborn
iris_df = sns.load_dataset('iris')
iris_df.head()

np.random.seed(1)
iris_hf_df = iris_df.copy()
K = 150
X_new_cols = []
for k in range(K):
    new_col_names = [f"petal_width_{k}", f"sepal_length_{k}", f"sepal_width_{k}"]
    iris_hf_df[new_col_names] = (
        iris_hf_df[["petal_width", "sepal_length", "sepal_width"]] @ np.random.rand(3, 3)
        + (np.random.rand(len(iris_hf_df), 3) - 0.5) * 30
    )
    X_new_cols.append(new_col_names)
    
iris_hf_df['petal_length'] = iris_hf_df[["petal_width", "sepal_length", "sepal_width"]] @ np.random.rand(3, 1)
X_new_cols = sum(X_new_cols, [])

# Set up INS/OOS sample data
np.random.seed(1)
ins_frac = 0.1
iris_ins_df = iris_hf_df.sample(70).reset_index(drop = True)
iris_oos_df = iris_hf_df.sample(70).reset_index(drop = True)

Y_ins = iris_ins_df[['petal_length']]
X_ins = iris_ins_df[X_new_cols]
Y_oos = iris_oos_df[['petal_length']]
X_oos = iris_oos_df[X_new_cols]

### Estimate

# Component algos
model_forecast_lasso = LASSO({"lambda": 1e-5, "use_intercept": False, "seed": 5}, n_iter=12)
model_forecast_enet = ENet(
    {"lambda": 1e-5, "l1_ratio": 0.5, "use_intercept": True, "seed": 5}, n_iter=14
)
model_forecast_ols = LinearRegression({})
model_selection_pca = PCA_selection({})
model_selection_lasso = LASSO_selection({})

# Post-selection algos
model_pca_lasso = PostSelectionModel(model_selection_pca, model_forecast_lasso, n_iter=58)
model_lasso_ols = PostSelectionModel(model_selection_lasso, model_forecast_ols)
model_pca_enet = PostSelectionModel(model_selection_pca, model_forecast_enet)
model_list = [model_forecast_lasso, model_pca_lasso, model_lasso_ols, model_pca_enet]

def helper_func(t):
    
    # Split into train/validate for optimal hyperparams
    mtrain = ModelTrainer(model_list, Y_ins, X_ins, seed=444)
    mtrain.validation(frac=0.5, n_iter_default=100)

    # Testing
    mtest = ModelTester(mtrain)
    model_forecasts, model_params = mtest.forecast(Y_oos, X_oos)
    
    return model_forecasts, model_params
    
import multiprocessing

# with multiprocessing.Pool(12) as p:

p = Pool(2)

for result in p.imap_unordered(helper_func, range(10)):
    logging.info('Completed iteration!' + '\n ================================')
    model_forecasts, model_params = result

p.close()
    
### Export

model_forecasts[0].to_csv(f'../../data/output/test_output_0.csv')

logging.info('Done!')

