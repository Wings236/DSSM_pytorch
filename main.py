import pandas as pd
import torch
from tqdm import tqdm
from utils import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from model import DSSM
from torch import optim, nn
from dataloader import Movie_data
from torch.utils.data import DataLoader

data = pd.read_csvdata = pd.read_csv("./data/movielens_sample.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres"]   # 对这些进行一个稀疏化
SEQ_LEN = 50
negsample = 10

# 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

# 数据分类与数据清洗
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id", "genres"]].drop_duplicates('movie_id')
user_profile.set_index("user_id", inplace=True)
# user_item_list = data.groupby("user_id")['movie_id'].apply(list)    # 通过user来进行分类，但是似乎没有用到
train_set, test_set = gen_data_set(data, SEQ_LEN, negsample)        # 负采样同时按照时间序列划分数据集
train_X, train_y = gen_model_input(train_set, user_profile, SEQ_LEN)
test_X, test_y = gen_model_input(test_set, user_profile, SEQ_LEN)

# 2.count #unique features for each sparse field and generate feature config for sequence feature
from utils import SparseFeat, VarLenSparseFeat
embedding_dim = 32

# 针对每一个变量做一个属性上的定义，同时给每一个变量一个维数进行表示
user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                        SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                        SparseFeat("age", feature_max_idx['age'], embedding_dim),
                        SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                        SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        VarLenSparseFeat(SparseFeat('hist_genres', feature_max_idx['genres'], embedding_dim,
                                                    embedding_name="genres"), SEQ_LEN, 'mean', 'hist_len'),
                        ]
item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim),
                        SparseFeat('genres', feature_max_idx['genres'], embedding_dim)
                        ]

train_dataset = Movie_data(train_X, train_y, user_feature_columns, item_feature_columns)
test_dataset = Movie_data(test_X, test_y, user_feature_columns, item_feature_columns)
trian_len = len(train_dataset)
test_len = len(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

model = DSSM(user_feature_columns, item_feature_columns)
lr = 0.001
l2_coff = 0.0
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coff)

num_epoch = 30
for epoch in range(num_epoch):
    # 训练
    model.train()
    train_loss = 0.0
    for X_user, X_item, y in tqdm(train_loader, ncols=80):
        optimizer.zero_grad()
        y_hat = model(X_user, X_item)
        loss = loss_function(y_hat, y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()/trian_len
    print(f'{epoch+1}, trian loss:{train_loss}')

    # 验证
    model.eval()
    test_loss = 0.0
    for X_user, X_item, y in tqdm(test_loader, ncols=80):
        y_hat = model(X_user, X_item)
        with torch.no_grad():
            loss = loss_function(y_hat, y.float())
        test_loss += loss.cpu().item()/test_len
    
    print(f'{epoch+1}, test loss:{test_loss}')