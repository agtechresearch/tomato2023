#%%

import sklearn
from tsai.basics import *
my_setup(sklearn)


#%%
import pandas as pd

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("merged.csv", index_col=0)[:500]
df["MSRM_DT"] = pd.to_datetime(df["MSRM_DT"], format="%Y-%m-%d %H:%M:%S")
# X_cols = pd.read_csv("control.csv", index_col=0).columns
X_cols = [ "EXTN_TPRT", "DWP_TPRT", "ABSLT_HMDT", "WDSP", "INNER_TPRT_1", "INNER_TPRT_2"]
y_cols = ["FRT_LNGTH"] # "FWRCT_HGHT"]
X_raw, y_raw = df[X_cols], df[y_cols]

#%%
datetime_col = "MSRM_DT"
fcst_history = 13 # # steps in the past
fcst_horizon = 1  # # steps in the future
valid_size   = 0.1  # int or float indicating the size of the training set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_forecasting_splits(df, fcst_history=fcst_history, 
                                fcst_horizon=fcst_horizon, datetime_col=datetime_col,
                                valid_size=valid_size, test_size=test_size)
splits
# %%

train_split = splits[0]

# pipeline
exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=X_cols)), # standardize data using train_split
    ], 
    verbose=True)
save_object(exp_pipe, 'data/exp_pipe.pkl')
exp_pipe = load_object('data/exp_pipe.pkl')

df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)
df_scaled.head()

# %%

X, y = prepare_forecasting_data(df, fcst_history=fcst_history, 
                                fcst_horizon=fcst_horizon, x_vars=X_raw, y_vars=y_raw)
X.shape, y.shape
# %%

arch_config = dict(
    n_layers=1,  # number of encoder layers
    n_heads=1,  # number of heads
    d_model=1,  # dimension of model
    d_ff=1,  # dimension of fully connected network
    attn_dropout=0.0, # dropout applied to the attention weights
    # dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
    patch_len=1,  # length of the patch applied to the time series to create patches
    stride=1,  # stride used when creating patches
    padding_patch=True,  # padding_patch
)

learn = TSForecaster(X, y, splits=splits, batch_size=1, 
                     path="models", pipelines=[exp_pipe],
                     arch="PatchTST", 
                    #  arch_config=arch_config, 
                     metrics=[mse, mae], cbs=ShowGraph())
learn.dls.valid.drop_last = True
learn.summary()
# %%
n_epochs = 100
lr_max = 0.0025
learn.fit_one_cycle(n_epochs, lr_max=lr_max)
learn.export('patchTST.pt')
# %%

from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error

learn = load_learner('models/patchTST.pt')
scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
scaled_preds = to_np(scaled_preds)
print(f"scaled_preds.shape: {scaled_preds.shape}")

scaled_y_true = y[splits[1]]
results_df = pd.DataFrame(columns=["mse", "mae"])
results_df.loc["valid", "mse"] = mean_squared_error(
    scaled_y_true.flatten(), scaled_preds.flatten())
results_df.loc["valid", "mae"] = mean_absolute_error(
    scaled_y_true.flatten(), scaled_preds.flatten())
results_df
#%%
from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error

learn = load_learner('models/patchTST.pt')
y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
y_test_preds = to_np(y_test_preds)
print(f"y_test_preds.shape: {y_test_preds.shape}")

y_test = y[splits[2]]
results_df = pd.DataFrame(columns=["mse", "mae"])
results_df.loc["test", "mse"] = mean_squared_error(
    y_test.flatten(), y_test_preds.flatten())
results_df.loc["test", "mae"] = mean_absolute_error(
    y_test.flatten(), y_test_preds.flatten())
results_df.head()
#%%

X_test = X[splits[2]]
y_test = y[splits[2]]
plot_forecast(X_test, y_test, y_test_preds, sel_vars=True)
# %%
fcst_date = "2023-03-30"
dates = pd.date_range(start=None, end=fcst_date, periods=fcst_history, freq="5m")
new_df = df[y_cols + X_cols].copy()
new_df = new_df[new_df[datetime_col].isin(dates)].reset_index(drop=True)
new_df.head()

#%%
from tsai.inference import load_learner

learn = load_learner('models/patchTST.pt')
new_df = learn.transform(new_df)
new_df.head()

#%%
new_X, _ = prepare_forecasting_data(new_df, 
                                    fcst_history=fcst_history, fcst_horizon=0, 
                                    x_vars=X, y_vars=None)
new_X.shape
#%%

new_scaled_preds, *_ = learn.get_X_preds(new_X)

new_scaled_preds = to_np(new_scaled_preds).swapaxes(1,2).reshape(-1, len(y))
dates = pd.date_range(start=fcst_date, periods=fcst_horizon + 1, freq='7D')[1:]
preds_df = pd.DataFrame(dates, columns=[datetime_col])
preds_df.loc[:, y] = new_scaled_preds
preds_df = learn.inverse_transform(preds_df)
preds_df