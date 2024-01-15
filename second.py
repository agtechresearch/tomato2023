#%%
import pandas as pd
import statsmodels.api as sm 
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("merged.csv", index_col=0)[:100]
X_cols = pd.read_csv("control.csv", index_col=0).columns

betas = {}

for y_col in pd.read_csv("growth.csv", index_col=0).columns:
    y_cols = [y_col]
    X_raw, y_raw = df[X_cols], df[y_cols]

    # 회귀분석을 하기 위한 B_0, 상수항 추가
    x_data1 = sm.add_constant(X_raw, has_constant = "add")

    # 회귀모델 적합
    multi_model = sm.OLS(y_raw, x_data1)
    fitted_multi_model = multi_model.fit()

    # summary함수를 통해 OLS 결과 출력
    fitted_multi_model.summary()
    #%%
    import matplotlib.pyplot as plt
    pred4 = fitted_multi_model.predict(x_data1)

    # residual plot 구하기
    # fitted_multi_model.resid.plot()
    # plt.xlabel("residual_number")
    # plt.show()
    betas[y_col] = fitted_multi_model.params
y_cols = ["FRT_LNGTH"] # "FWRCT_HGHT"]
X_raw, y_raw = df[X_cols], df[y_cols]


# %%
# 회귀분석을 하기 위한 B_0, 상수항 추가
x_data1 = sm.add_constant(X_raw, has_constant = "add")

# 회귀모델 적합
multi_model = sm.OLS(y_raw, x_data1)
fitted_multi_model = multi_model.fit()

# summary함수를 통해 OLS 결과 출력
fitted_multi_model.summary()
#%%
import matplotlib.pyplot as plt
pred4 = fitted_multi_model.predict(x_data1)

# residual plot 구하기
fitted_multi_model.resid.plot()
plt.xlabel("residual_number")
plt.show()
print(fitted_multi_model.params)