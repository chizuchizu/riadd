import pandas as pd
from tqdm import tqdm
from glob import glob
import numpy as np

df = pd.read_csv("../extra/trainLabels.csv")
train = pd.read_csv("../data/Training_Set/RFMiD_Training_Labels.csv")
img_list = glob("../extra/train/*.jpeg")
use_idx = []

# 異常なし(level 0)のデータのみ
for x in tqdm(img_list):
    img_id = x.replace(".jpeg", "").replace("../extra/train/", "")
    memo = df[df["image"] == img_id]
    level = memo["level"].iloc[0]

    if level == 0:
        use_idx.append(memo.index[0])

use_df = df.iloc[use_idx, :]
memo = np.zeros(
    (
        use_df.shape[0],
        train.shape[1]
    ),
    dtype=np.uint8
)

memo = pd.DataFrame(
    memo,
    columns=train.columns
)
memo.iloc[:, 0] = use_df["image"].values
memo.to_csv("../extra/use_df.csv", index=False)
