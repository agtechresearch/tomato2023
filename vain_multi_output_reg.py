#%%
import pandas as pd

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("merged.csv", index_col=0)
X_cols = pd.read_csv("control.csv", index_col=0).columns
y_cols = ["FRT_LNGTH", "FWRCT_HGHT"]

#%% # returns a Bunch instance
X, y = df[X_cols], df[y_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


pred = MultiOutputRegressor(
    GradientBoostingRegressor(random_state=0)
).fit(X_train, y_train).predict(X_test)

#%%

print("----------","Total","----------")
print(mean_squared_error(y_test, pred))
print(mean_absolute_error(y_test, pred))
print(r2_score(y_test, pred))

for i, v in enumerate(y_cols):
    print("----------", i, v, "----------")
    print(mean_squared_error(y_test[v], pred[:, i]))
    print(mean_absolute_error(y_test[v], pred[:, i]))
    print(r2_score(y_test[v], pred[:, i]))

for i, v in enumerate(y_cols):
    pred = GradientBoostingRegressor(
        random_state=0
    ).fit(X_train, y_train[v]).predict(X_test)

    print("----------", i, v, "----------")
    print(mean_squared_error(y_test[v], pred))
    print(mean_absolute_error(y_test[v], pred))
    print(r2_score(y_test[v], pred))

'''
---------- Total ----------
168.30721068226546
7.3490584341315515
0.7921588186325197
---------- 0 FRT_LNGTH ----------
3.366203315974518
1.3178197367693494
0.796278081362803
---------- 1 FWRCT_HGHT ----------
333.2482180485564
13.380297131493753
0.7880395559022363
---------- 0 FRT_LNGTH ----------
3.366203315974518
1.3178197367693494
0.796278081362803
---------- 1 FWRCT_HGHT ----------
333.2482180485564
13.380297131493753
0.7880395559022363
'''
# %%

from xgboost import XGBRegressor

# Define and train the model
model = XGBRegressor(tree_method='hist', random_state=0)


pred = MultiOutputRegressor(
    model
).fit(X_train, y_train).predict(X_test)

#%%

print("Total")
print(mean_squared_error(y_test, pred))
print(mean_absolute_error(y_test, pred))
print(r2_score(y_test, pred))

for i, v in enumerate(y_cols):
    print("----------", i, v, "----------")
    print(mean_squared_error(y_test[v], pred[:, i]))
    print(mean_absolute_error(y_test[v], pred[:, i]))
    print(r2_score(y_test[v], pred[:, i]))

for i, v in enumerate(y_cols):
    pred = model.fit(X_train, y_train[v]).predict(X_test)
    print("----------", i, v, "----------")
    print(mean_squared_error(y_test[v], pred))
    print(mean_absolute_error(y_test[v], pred))
    print(r2_score(y_test[v], pred))

'''
Total
16.01488539107893
1.9678512160550978
0.9792015080662064
---------- 0 FRT_LNGTH ----------
0.35443087306247373
0.37454467699010524
0.9785499179024959
---------- 1 FWRCT_HGHT ----------
31.675339909095385
3.5611577551200906
0.979853098229917
---------- 0 FRT_LNGTH ----------
0.35443087306247373
0.37454467699010524
0.9785499179024959
---------- 1 FWRCT_HGHT ----------
31.675339909095385
3.5611577551200906
0.979853098229917
'''    

