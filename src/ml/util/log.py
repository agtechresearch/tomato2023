import sys 
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def describe_df(df):
    print(df.describe())
    df.plot(legend=True, figsize=(15, 5))
    plt.suptitle('Original Dataset')
    plt.show()

def plot_ts_result(true_val, pred_val, bonus=None, fname="plot_result.png"):
    plt.figure(figsize=(15, 5)) 
    plt.plot(true_val, linestyle='solid')
    plt.plot(pred_val, linestyle='dotted')
    if bonus:
        plt.plot(bonus, linestyle="dotted")
    plt.suptitle('Time-Series Prediction')
    plt.savefig(fname)
    plt.show()
    


def parameter_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('parameter size: {:.6f}MB'.format(size_all_mb))
    print('model dict size:', sys.getsizeof(model.state_dict()))

def save_model(model, fname="model", fpath="model_file"):
    parameter_size(model)
    torch.save(model.state_dict(), f"./{fpath}/{fname}.pt")

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))