import pandas as pd
import numpy as np

train = pd.read_csv("../data/Training_Set/RFMiD_Training_Labels.csv")

sub_class = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 3, 0, 0, 0, 3, 3, 3, 3, 2, 0, 0, 1, 0, 0, 1, 2, 0]

classes = np.zeros(
    (
        len(train),
        4
    )
)

for i, x in enumerate(sub_class):
    memo = train.iloc[:, i + 2]
    classes[:, x] += memo

classes = np.where(classes > 0, 1, 0)
train["c_0"] = classes[:, 0]
train["c_1"] = classes[:, 1]
train["c_2"] = classes[:, 2]
train["c_3"] = classes[:, 3]

train.to_csv("../data/Training_Set/train_class.csv", index=False)
