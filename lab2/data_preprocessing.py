import os
import glob
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROC_DIR, exist_ok=True)

train_dfs = []
test_dfs = []

for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    df = pd.read_csv(path)
    if "_test" in os.path.basename(path):
        test_dfs.append(df)
    else:
        train_dfs.append(df)

unified_train = pd.concat(train_dfs, ignore_index=True)
unified_test  = pd.concat(test_dfs, ignore_index=True)

unified_train.to_csv(os.path.join(PROC_DIR, "unified_train.csv"), index=False)
unified_test.to_csv(os.path.join(PROC_DIR, "unified_test.csv"), index=False)