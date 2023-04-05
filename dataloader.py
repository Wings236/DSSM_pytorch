from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
class Movie_data(Dataset):
    def __init__(self, X, y, user_feature_columns, item_feature_columns, mapping_idx):
        super(Movie_data, self).__init__()
        X_user = {feature.name:X[feature.name] for feature in user_feature_columns}
        X_item = {feature.name:X[feature.name] for feature in item_feature_columns}

        def to_list(x):
            res = []
            for key in x.keys():
                name = key.lstrip('hist_')
                if isinstance(x[key], list):
                    res.extend((np.array(x[key]) + mapping_idx[name]).tolist())
                else:
                    res.append(x[key] + mapping_idx[name])
            return res
        # user
        user_pd = pd.DataFrame(X_user)
        data_X_user = user_pd.apply(to_list, axis=1).values.tolist()
        # item
        user_pd = pd.DataFrame(X_item)
        data_X_item = user_pd.apply(to_list, axis=1).values.tolist()

        self.X_user = torch.tensor(data_X_user)
        self.X_item = torch.tensor(data_X_item)
        self.y = torch.tensor(y)
    
    def __getitem__(self, index):
        return self.X_user[index], self.X_item[index], self.y[index]
    
    def __len__(self):
        return len(self.X_user)
