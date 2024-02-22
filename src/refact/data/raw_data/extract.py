#%%
from rich import print
import os
import pandas as pd
import numpy as np


#%%
path = "./november"
data = [pd.read_csv(f"{path}/{file}") for file in os.listdir(path)]
print("Num of files", len(data))

#%%
data[0].iloc[0]

#%%
cols = {}
col_nums = []
for d in data:
    del d["ZONE_NM"]
    col_nums.append(len(d.columns))
    if d is not None:
        for col in d.columns:
            if col not in cols:
                cols[col] = 1
            else:
                cols[col] += 1

print(cols)

ample_file = os.listdir(path)[np.argmax(col_nums)]
print("Maxium number of columns", ample_file)

#%%
## 모든 파일에 들어있는 컬럼들이 모두 같은 값을 가지는지 확인
columns_all = [k for k in cols.copy() if cols[k] == 15]
check_all_same = np.zeros(len(data)-1)
for i in range(1, len(data)):
    check_all_same[i-1] = \
        ((data[i][columns_all] == data[i-1][columns_all]).all()).all()
print(check_all_same.all())

#%%
# for k in cols.copy():
#     if 1 < cols[k] < 14:
#         print(k, cols[k])
#         del cols[k]

# 환경정보이나 모든 파일에 들어있지 않아서 제거
# for k in ["SUB_MHRLS_OPRT_YN_4", "CLR_OPRT_YN_3", "HTNG_TPRT_2"]:
#     del cols[k]

print(cols)

#%%
growth_key_name = [
    "FRT_LNGTH", "FWRCT_HGHT", "BLMNG_CLUSTER", "FRST_TREE_CNT",
    "FRT_WT", "FRT_WDTH", "GRTH_LNGTH", "YIELD_CNT", "YIELD_CLUSTER",
    "LAST_FWRCT_NO", "LEAF_LNGTH", "LEAF_CNT", "LEAF_WDTH", "PLT_LNGTH", "STEM_THNS"
]
growth_file_name = [
    "TOMATO_FRUIT_LEN_ENV_20231123.csv", "TOMATO_FLOWER_CLUSTER_HEIGHT_ENV_20231123.csv",
    "TOMATO_FLOWER_PER_TRUSS_ENV_20231123.csv", "TOMATO_FRUIT_SETTING_ENV_20231123.csv",
    "TOMATO_FRUIT_WEIGHT_ENV_20231123.csv", "TOMATO_FRUIT_WIDTH_ENV_20231123.csv",
    "TOMATO_GROWTH_LENGTH_ENV_20231123.csv", "TOMATO_HARVEST_ENV_20231123.csv",
    "TOMATO_HARVEST_PER_TRUSS_ENV_20231123.csv", "TOMATO_LAST_FLOWERING_BUD_ENV_20231123.csv",
    "TOMATO_LEAF_LEN_ENV_20231123.csv", "TOMATO_LEAF_NUM_ENV_20231123.csv",
    "TOMATO_LEAF_WIDTH_ENV_20231123.csv", "TOMATO_SOIL_SURFACE_LEN_ENV_20231123.csv",
    "TOMATO_STEM_THICKNESS_ENV_20231123.csv"
]


growth = []
control = []
for k, v in cols.items():
    if v == 1 and k in growth_key_name:
        growth.append(k)
    else:
        control.append(k)
    

new_df = {}
for i in range(len(growth_key_name)):
    new_df[growth_key_name[i]] = pd.read_csv(f"{path}/{growth_file_name[i]}",
                              usecols=[growth_key_name[i]])[growth_key_name[i]]
    
new_df = pd.DataFrame(new_df)

control.remove("HTNG_TPRT_2")
new_df = pd.concat([new_df, pd.read_csv(f"{path}/TOMATO_STEM_THICKNESS_ENV_20231123.csv", usecols=["HTNG_TPRT_2"])], axis=1)
control.remove("CLR_OPRT_YN_3")
new_df = pd.concat([new_df, pd.read_csv(f"{path}/TOMATO_LEAF_NUM_ENV_20231123.csv", usecols=["CLR_OPRT_YN_3"])], axis=1)


new_df = pd.concat([new_df, pd.read_csv(f"{path}/{ample_file}", usecols=control)], axis=1)
print(new_df.shape, len(growth_key_name)+len(control))
#%%

# 모든 일자가 5분 간격으로 끊김없이 연속적으로 존재하는지 확인
from datetime import datetime, timedelta
for t in range(1, new_df["MSRM_DT"].shape[0]):
    before = datetime.strptime( new_df["MSRM_DT"].iloc[t-1], "%Y-%m-%d %H:%M:%S")+timedelta(minutes=5)
    after = datetime.strptime( new_df["MSRM_DT"].iloc[t], "%Y-%m-%d %H:%M:%S")
    if before != after:
        print(t, before, after)
new_df.head()
#%%

#%%
# 해당 컬럼의 모든 값이 동일하면 제거
print(new_df.shape)
for col in new_df.columns:
    if new_df[col].nunique() == 1:
        print(col, new_df[col].unique())
        del new_df[col]
print(new_df.shape)
new_df.head()
#%%

import sweetviz as sv
new_df.to_csv("merged_null.csv", index=False)
my_report = sv.analyze(new_df)
my_report.show_html("sweetviz_merged_null.html")

new_df.ffill(inplace=True)
new_df.bfill(inplace=True)
new_df.to_csv("merged_fill.csv", index=False)
new_df.info()

my_report = sv.analyze(new_df)
my_report.show_html("sweetviz_merged_fill.html")
# Open with Live server in vscode로 열면 확인 가능 (확장 프로그램 설치 필요)
# %%
