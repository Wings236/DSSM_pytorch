import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from utils import gen_data_set, gen_model_input, SparseFeat, VarLenSparseFeat, Precision, set_seed
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from model import DSSM
from torch import optim, nn
from dataloader import Movie_data
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = pd.read_csvdata = pd.read_csv("./data/movielens_sample.txt")
data = pd.read_csvdata = pd.read_csv("./data/ml-100k.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres"]   # 对这些进行一个稀疏化
SEQ_LEN = 30
negsample = 5
embedding_dim = 32
seed = 2023
set_seed(seed)

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
train_len = len(train_dataset)
test_len = len(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True) #? 计算的瓶颈在于模型本身对数据的操作
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

lr = 1e-4
l2_coff = 1e-5
loss_function = nn.BCELoss()
model = DSSM(user_feature_columns, item_feature_columns).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coff)

num_epoch = 30
early_stopping = 5
patience = 0
best_test_loss = np.inf

# 可以把CTR看做是二分类，看哪个概率大做哪个行为

for epoch in range(num_epoch):
    # 训练
    model.train()
    train_pred_y = []
    train_true_y = []
    train_prec = 0.0
    train_loss, train_prec = 0.0, 0.0
    for X_user, X_item, y in tqdm(train_loader, ncols=80):
        optimizer.zero_grad()
        y_hat = model(X_user, X_item)
        loss = loss_function(y_hat.cpu(), y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()/train_len
        train_pred_y.extend(y_hat.cpu().detach().numpy())
        train_true_y.extend(y.numpy())
        train_prec+= Precision(y_hat.cpu(), y)/train_len
        # print(train_pred_y, train_true_y)
    train_AUC = roc_auc_score(train_true_y, train_pred_y)

    # 验证
    model.eval()
    test_pred_y = []
    test_true_y = []
    test_prec = 0.0
    test_loss, test_prec = 0.0, 0.0
    for X_user, X_item, y in tqdm(test_loader, ncols=80):
        y_hat = model(X_user, X_item)
        with torch.no_grad():
            loss = loss_function(y_hat.cpu(), y.float())
        test_loss += loss.cpu().item()/test_len
        test_pred_y.extend(y_hat.cpu().detach().numpy())
        test_true_y.extend(y)
        test_prec+= Precision(y_hat.cpu(), y)/test_len
    test_AUC = roc_auc_score(test_true_y, test_pred_y)

    if best_test_loss > test_loss:
        print('find the better result')
        best_test_loss = test_loss
        best_prec = test_prec
        best_auc = test_AUC
        patience = 0
    print(f'EPOCH:{epoch+1}({patience}/{early_stopping})')
    print(f'trian loss:{train_loss:.4f}, precision:{train_prec:.4f}, AUC:{train_AUC:.4f}')
    print(f'test loss:{test_loss:.4f}, precision:{test_prec:.4f}, AUC:{test_AUC:.4f}')
    if patience >= early_stopping:
        print('res will not be better, training is done.')
        print(f'the best res is, loss :{best_test_loss:.4f}, precision:{best_prec:.4f}, AUC:{best_auc:.4f}')
        break
    
    patience += 1



    