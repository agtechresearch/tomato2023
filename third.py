#%%
from tsai.basics import *
import pandas as pd
import statsmodels.api as sm 
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("merged.csv", index_col=0)
X_cols = pd.read_csv("control.csv", index_col=0).columns
X_raw = df[X_cols]
y_cols = ["FRT_LNGTH"] 
X, y = df[X_cols].to_numpy(), df[y_cols].to_numpy()

datetime_col = "MSRM_DT"
fcst_history = 13 # # steps in the past
fcst_horizon = 1  # # steps in the future
valid_size   = 0.1  # int or float indicating the size of the training set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_forecasting_splits(df, fcst_history=fcst_history, 
                                fcst_horizon=fcst_horizon, datetime_col=datetime_col,
                                valid_size=valid_size, test_size=test_size)

# X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)
X.shape, y.shape, splits
#%%
tfms = [None, TSRegression()]
batch_tfms = TSStandardize(by_sample=True)
reg = TSRegressor(X, y, splits=splits, path='models', arch="TSTPlus", tfms=tfms, batch_tfms=batch_tfms, metrics=rmse, cbs=ShowGraph(), verbose=True)
reg.fit_one_cycle(100, 3e-4)
reg.export("reg.pkl")


from tsai.inference import load_learner

reg = load_learner("models/reg.pkl")
raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])







# %%
from tsai.basics import *

ts = get_forecasting_time_series("Sunspots").values
X, y = SlidingWindow(60, horizon=1)(ts)
splits = TimeSplitter(235)(y) 
tfms = [None, TSForecasting()]
batch_tfms = TSStandardize()
fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=512, arch="TSTPlus", metrics=mae, cbs=ShowGraph())
fcst.fit_one_cycle(50, 1e-3)
fcst.export("fcst.pkl")


from tsai.inference import load_learner

fcst = load_learner("models/fcst.pkl", cpu=False)
raw_preds, target, preds = fcst.get_X_preds(X[splits[1]], y[splits[1]])
raw_preds.shape
# torch.Size([235, 1])
# %%
