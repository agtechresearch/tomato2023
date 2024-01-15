#%%
import pandas as pd

growth_df = pd.read_csv("growth.csv", index_col=0)
control_df = pd.read_csv("control.csv", index_col=0)

#%%
growth_df.head()
# %%
growth_df.describe()
# %%
control_df.head()
#%%
control_df.describe()

# %%

df = pd.merge(growth_df, control_df, how="inner", on="MSRM_DT")
print(df.shape, growth_df.shape, control_df.shape)
# df.ffill(inplace=True, limit=3)
# df.bfill(inplace=True, limit=3)

df["MSRM_DT"] = pd.to_datetime(df["MSRM_DT"], format="%Y-%m-%d %H:%M:%S")
df.dropna(inplace=True)
print(df.shape, growth_df.shape, control_df.shape)
df.sort_values(by="MSRM_DT", inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()
# %%
df.tail()
# %%
df.info()
# %%
df.describe()
# %%
# Check for duplicates in dates
# df["MSRM_DT"].value_counts().sort_values(ascending=False)
#%%
df.to_csv("merged.csv", index=False)
# %%
