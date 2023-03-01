from torch.utils.data import Dataset
class Movie_data(Dataset):
    def __init__(self, X, y, user_feature_columns, item_feature_columns):
        super(Movie_data, self).__init__()
        X_user = {feature.name:X[feature.name] for feature in user_feature_columns}
        X_item = {feature.name:X[feature.name] for feature in item_feature_columns}
        data_X_user = []
        for value in X_user.values():
            for idx, data in enumerate(value):
                try:
                    data_X_user[idx].append(data)
                except:
                    data_X_user.append([data])
        
        data_X_item = []
        for value in X_item.values():
            for idx, data in enumerate(value):
                try:
                    data_X_item[idx].append(data)
                except:
                    data_X_item.append([data])
        self.X_user = data_X_user
        self.X_item = data_X_item
        self.y = y
    
    def __getitem__(self, index):
        return self.X_user[index], self.X_item[index], self.y[index]
    
    def __len__(self):
        return len(self.X_user)
