import pandas as pd
import torch
from tqdm import tqdm
from utils import gen_data_set, gen_model_input, SparseFeat, VarLenSparseFeat, set_seed, Annoy, Test
from sklearn.preprocessing import LabelEncoder
from model import DSSM
from torch import optim, nn
from dataloader import Movie_data
from torch.utils.data import DataLoader

cpu_id = 4
device = torch.device(f'cuda:{cpu_id}' if torch.cuda.is_available() else 'cpu')
# data = pd.read_csvdata = pd.read_csv("./data/movielens_sample.txt")
data = pd.read_csvdata = pd.read_csv("./data/ml-100k.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres"]
SEQ_LEN = 50
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
all_item = [item_profile.values[:,idx] for idx in range(len(item_feature_columns))]

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True) #? 计算的瓶颈在于模型本身对数据的操作
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

lr = 1e-3
l2_coff = 1e-5
drop_rate = 0.25
loss_function = nn.BCELoss()
model = DSSM(user_feature_columns, item_feature_columns, dnn_dropout=drop_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coff)

num_epoch = 30
topks = [20, 50]
early_stopping = 5
patience = 0
best_test_recall = 0
# 换成召回模式

for epoch in range(num_epoch):
    # 训练
    model.train()
    train_loss = 0.0
    for X_user, X_item, y in tqdm(train_loader, ncols=80):
        optimizer.zero_grad()
        y_hat = model(X_user, X_item)
        loss = loss_function(y_hat.cpu(), y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()/train_len

    # 验证
    model.eval()
    annoy = Annoy('angular')
    item_embedding = model.item_embedding(all_item).cpu()
    annoy.item_embedding_store(all_item, item_embedding)
    pred_y, test_y = [], []
    for X_user, X_item, y in tqdm(test_loader, ncols=80):
        # 针对每一个X_user进行搜索
        with torch.no_grad():
            user_embedding = model.user_mebedding(X_user).cpu()
        pred_y.extend([annoy.search_item(user_emb, top_k=max(topks)) for user_emb in user_embedding])
        test_y.extend([[item_id]for item_id in X_item[0].numpy()])
    # 得到每一个指标
    res = Test(pred_y, test_y, topks=topks)    # 计算对应的指标
    for temp_res in res:
        print(temp_res)
        
    if best_test_recall < res[-1][f'recall@{topks[-1]}']:
        print('find the better result')
        best_res = res
        best_test_recall = res[-1][f'recall@{topks[-1]}']
        patience = 0
    
    print(f'EPOCH:{epoch+1}({patience}/{early_stopping})')
    print(f'trian loss:{train_loss:.4f}')
    for idx, topk in enumerate(topks):
        print(f'recall@{topk}:{res[idx][f"recall@{topk}"]}, ndcg@{topk}:{res[idx][f"ndcg@{topk}"]}')

    if patience >= early_stopping:
        print('res will not be better, training is done.')
        print(f'the best res is')
        for temp_res in best_res:
            print(temp_res)
        break
    
    patience += 1



    