from torch.utils.data import Dataset
import numpy as np
import torch
class Movie_data(Dataset):
    def __init__(self, X, y, user_feature_columns, item_feature_columns, mapping_idx):
        super(Movie_data, self).__init__()
        X_user = {feature.name:X[feature.name] for feature in user_feature_columns}
        X_item = {feature.name:X[feature.name] for feature in item_feature_columns}

        data_X_user = []
        for name, value in X_user.items():
            name = name.lstrip('hist_')
            for idx, data in enumerate(value):
                try:
                # 判断是不是列表
                    if isinstance(data, np.ndarray):
                        for temp_seq_data in data:
                            data_X_user[idx].append(temp_seq_data + mapping_idx[name])
                    else:
                        data_X_user[idx].append(data + mapping_idx[name])
                except:
                    if isinstance(data, np.ndarray):
                        for temp_seq_data in data:
                            try:
                                data_X_user[idx].append(temp_seq_data + mapping_idx[name])
                            except:
                                data_X_user.append([temp_seq_data + mapping_idx[name]])
                    else:
                        data_X_user.append([data + mapping_idx[name]])


        
        data_X_item = []
        for name, value in X_item.items():
            name = name.lstrip('hist_')
            for idx, data in enumerate(value):
                try:
                # 判断是不是列表
                    if isinstance(data, np.ndarray):
                        for temp_seq_data in data:
                            data_X_item[idx].append(temp_seq_data + mapping_idx[name])
                    else:
                        data_X_item[idx].append(data + mapping_idx[name])
                except:
                    if isinstance(data, np.ndarray):
                        for temp_seq_data in data:
                            try:
                                data_X_item[idx].append(temp_seq_data + mapping_idx[name])
                            except:
                                data_X_item.append([temp_seq_data + mapping_idx[name]])
                    else:
                        data_X_item.append([data + mapping_idx[name]])
        self.X_user = torch.tensor(data_X_user)
        self.X_item = torch.tensor(data_X_item)
        self.y = torch.tensor(y)
    
    def __getitem__(self, index):
        return self.X_user[index], self.X_item[index], self.y[index]
    
    def __len__(self):
        return len(self.X_user)
