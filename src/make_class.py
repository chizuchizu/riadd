import pandas as pd

train = pd.read_csv('../data/Training_Set/RFMiD_Training_Labels_all.csv')

map_dict = {
    "黄斑": 0,
    "視神経乳頭": 1,
    "全体的（色味）": 2,
    "真ん中": 3,
    "全体にみられる": 4
}

map_list = []
for i in range()