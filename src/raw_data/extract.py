#%%
from rich import print
import os
import pandas as pd


#%%
path = "./november"
data = [pd.read_csv(f"{path}/{file}") for file in os.listdir(path)]

#%%
data[0].iloc[0,2]

#%%
cols = {}
for d in data:
    if d is not None:
        print(d.shape)
        for col in d.columns:
            if col not in cols:
                cols[col] = 1
            else:
                cols[col] += 1
#%%
for k in cols.copy():
    if 1 < cols[k] < 14:
        print(k, cols[k])
        del cols[k]
print(cols)

# 환경정보이나 모든 파일에 들어있지 않아서 제거
for k in ["SUB_MHRLS_OPRT_YN_4", "CLR_OPRT_YN_3", "HTNG_TPRT_2"]:
    del cols[k]

del cols["ZONE_NM"] # 아마도 의미 없는 구역 정보

print(cols)

#%%
growth = []
control = []
for k, v in cols.items():
    if v >= 14:
        control.append(k)
    else:
        growth.append(k)

y = [
    ['FRT_LNGTH', "TOMATO_FRUIT_LEN_ENV_20231123.csv"],
    ['FWRCT_HGHT', "TOMATO_FLOWER_CLUSTER_HEIGHT_ENV_20231123.csv"],
    ['BLMNG_CLUSTER',"TOMATO_FLOWER_PER_TRUSS_ENV_20231123.csv"],
    ['FRST_TREE_CNT', "TOMATO_FRUIT_SETTING_ENV_20231123.csv"],
    ["FRT_WT", "TOMATO_FRUIT_WEIGHT_ENV_20231123.csv"],
    ["FRT_WDTH", "TOMATO_FRUIT_WIDTH_ENV_20231123.csv"],
    ["GRTH_LNGTH", "TOMATO_GROWTH_LENGTH_ENV_20231123.csv"],
    ["YIELD_CNT", "TOMATO_HARVEST_ENV_20231123.csv"],
    ["YIELD_CLUSTER", "TOMATO_HARVEST_PER_TRUSS_ENV_20231123.csv"],
    ["LAST_FWRCT_NO", "TOMATO_LAST_FLOWERING_BUD_ENV_20231123.csv"],
    ["LEAF_LNGTH", "TOMATO_LEAF_LEN_ENV_20231123.csv"],
    ["LEAF_CNT", "TOMATO_LEAF_NUM_ENV_20231123.csv"],
    ["LEAF_WDTH", "TOMATO_LEAF_WIDTH_ENV_20231123.csv"],
    ["PLT_LNGTH", "TOMATO_SOIL_SURFACE_LEN_ENV_20231123.csv"],
    ["STEM_THNS", "TOMATO_STEM_THICKNESS_ENV_20231123.csv"]
]

new_df = {}
for col, fname in y:
    new_df[col] = pd.read_csv(f"{path}/{fname}",
                              usecols=[col])[col]
    
new_df = pd.DataFrame(new_df)
new_df = pd.concat([new_df, pd.read_csv(f"{path}/{fname}", usecols=control)], axis=1)

# 모든 일자가 5분 간격으로 끊김없이 연속적으로 존재하는지 확인
from datetime import datetime, timedelta
for t in range(1, new_df["MSRM_DT"].shape[0]):
    before = datetime.strptime( new_df["MSRM_DT"].iloc[t-1], "%Y-%m-%d %H:%M:%S")+timedelta(minutes=5)
    after = datetime.strptime( new_df["MSRM_DT"].iloc[t], "%Y-%m-%d %H:%M:%S")
    if before != after:
        print(t, before, after)
new_df.head()
#%%
# 해당 컬럼의 모든 값이 동일하면 제거
print(new_df.shape)
for col in new_df.columns:
    if new_df[col].nunique() == 1:
        print(col, new_df[col].unique())
        del new_df[col]
print(new_df.shape)




#%%
new_df.to_csv("merged_all.csv", index=False)
new_df.dropna().to_csv("merged_drop.csv", index=False)
new_df.info()

# %%

import sweetviz as sv

my_report = sv.analyze(new_df)
my_report.show_html("sweetviz_merged_all.html")
# %%
