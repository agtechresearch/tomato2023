#%%
from rich import print
import os
import pandas as pd


#%%
data = [pd.read_csv(f"{file}") if file[-4:]==".csv" else None for file in os.listdir("data")]

#%%
data[0].iloc[0,2]

#%%
cols = {}
for d in data:
    if d:
        print(d.shape)
        for col in d.columns:
            if col not in cols:
                cols[col] = 1
            else:
                cols[col] += 1
print(cols)

for k, v in cols.items():
    if v == 1:
        print(f"'{k}',", end=" ")

# %%

y = [
    ['FRT_LNGTH', "TOMATO_FRUIT_LEN_ENV_20231123.csv"],
    ['FWRCT_HGHT', "TOMATO_FLOWER_CLUSTER_HEIGHT_ENV_20231123.csv"],
    ['BLMNG_CLUSTER',"TOMATO_FLOWER_PER_TRUSS_ENV_20231123.csv"],
    ['FRST_TREE_CNT', "TOMATO_FRUIT_SETTING_ENV_20231123.csv"],
    ["FRT_WT", "TOMATO_FRUIT_WEIGHT_ENV_20231123.csv"],
    ["FRT_WDTH", "TOMATO_FRUIT_WIDTH_ENV_20231123.csv"],
    ["GRTH_LNGTH", "TOMATO_GROWTH_LENGTH_ENV_20231123.csv"],
    # ["YIELD_CNT", "TOMATO_HARVEST_ENV_20231123.csv"],
    ["YIELD_CLUSTER", "TOMATO_HARVEST_PER_TRUSS_ENV_20231123.csv"],
    ["LAST_FWRCT_NO", "TOMATO_LAST_FLOWERING_BUD_ENV_20231123.csv"],
    ["LEAF_LNGTH", "TOMATO_LEAF_LEN_ENV_20231123.csv"],
    ["LEAF_CNT", "TOMATO_LEAF_NUM_ENV_20231123.csv"],
    ["LEAF_WDTH", "TOMATO_LEAF_WIDTH_ENV_20231123.csv"],
    ["PLT_LNGTH", "TOMATO_SOIL_SURFACE_LEN_ENV_20231123.csv"],
    ["STEM_THNS", "TOMATO_STEM_THICKNESS_ENV_20231123.csv"]
]


new_df = {
    "MSRM_DT": pd.read_csv("TOMATO_FRUIT_LEN_ENV_20231123.csv",
                           usecols=["MSRM_DT"])["MSRM_DT"]
}
for col, fname in y:
    new_df[col] = pd.read_csv(f"{fname}",
                              usecols=[col])[col]
    

new_df = pd.DataFrame(new_df)
new_df.head()

new_df.to_csv("growth.csv", index=False)

#%%

x = [
    'MSRM_DT', 
# 'ZONE_NM', 
'PFBS_NTRO_CBDX_CTRN', 
'EXTN_TPRT', 
'DWP_TPRT', 
'WNDRC', 
'ABSLT_HMDT', 
'WDSP', 
'STRTN_WATER', 
'EXTN_SRQT', 
'WATER_LACK_VL', 
'EXTN_ACCMLT_QOFLG', 
'SPL_TPRT_1', 
'SPL_TPRT_2', 
'HTNG_TPRT_1', 
'VNTILAT_TPRT_5', 
'VNTILAT_TPRT_4', 
'VNTILAT_TPRT_3', 
'VNTILAT_TPRT_2', 
'VNTILAT_TPRT_1', 
'TRWVLV_OPDR_RATE_2', 
'TRWVLV_OPDR_RATE_1', 
'HRZNT_SCRN_OPDR_RATE_2', 
'SKLT_OPDR_RATE_1_LEFT', 
'HRZNT_SCRN_OPDR_RATE_1', 
'SKLT_OPDR_RATE_1_RIGHT', 
'INNER_TPRT_1', 
'INNER_TPRT_2', 
'AVE_INNER_TPRT_1_2', 
'AVE_INNER_HMDT_1_2', 
'INNER_HMDT_1', 
'INNER_HMDT_2', 
'CBDX_STNG_VL', 
'WTSPL_QTY', 
'NTSLT_SPL_PH_LVL', 
'NTSLT_SPL_PH_LVL_STNG_VL', 
'NTSLT_SPL_ELCDT', 
'NTSLT_SPL_ELCDT_STNG_VL', 
'DYTM_NIGHT_CD', 
'SPRYN_DEVICE', 
'SUB_MHRLS_OPRT_YN_1', 
'SUB_MHRLS_OPRT_YN_2', 
'SUB_MHRLS_OPRT_YN_3', 
'SUB_MHRLS_OPMD_2', 
'PRCPT_YN', 
'FMGEQ_OPMD', 
'CBDX_GNRT_OPMD', 
'TRWVLV_OPMD_1', 
'SKLT_OPMD_1_LEFT', 
'FMGEQ_OPRT_YN', 
'CBDX_GNRT_OPRT_YN', 
]

new_df = pd.read_csv("TOMATO_FRUIT_LEN_ENV_20231123.csv",
                     usecols=x, index_col=0)
new_df.to_csv("control.csv", index=False)


#%%


for i in range(1, len(data)):
    for xx in x:
        print((data[i][x] == data[i-1][x]).all())
            
# %%
