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

for c in control_df.columns:
    print(c, control_df[c].mean())
# %%

CLR_OPRT_YN_2 0.0
CLR_OPMD_2 0.0
CLR_OPMD_3 0.0
CLR_OPMD_4 0.0
CLR_OPMD_5 0.0
FLFN_OPRT_YN 0.0
FLFN_OPMD 0.0
SKLT_OPMD_2_LEFT 0.0
BL_OPRT_YN 0.0