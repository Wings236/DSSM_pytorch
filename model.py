import torch
from utils import SparseFeat
from torch import nn
class DSSM(nn.Module):
    def __init__(self, user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_dropout=0):
        super(DSSM, self).__init__()
        '''Instantiates the Deep Structured Semantic Model architecture.
        :param user_feature_columns: An iterable containing user's features used by  the model.
        :param item_feature_columns: An iterable containing item's features used by  the model.
        :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
        :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
        :param dnn_activation: Activation function to use in deep net
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        '''
        # 参数初始化
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        total_feature = user_feature_columns + item_feature_columns
        # 不管你是变长还是其他，都可以通过取对应这个位置总数来充当对应的词向量大小
        self.embedding_dict = nn.ModuleDict({feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim) for feat in total_feature})
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.01)
        ## user
        # 每一个DNN的输入的维度应该是dim*Sparse个数+dim*maxlen*Varlen的个数
        user_DNN_input_dim = sum([feature.embedding_dim if isinstance(feature, SparseFeat) else feature.maxlen*feature.embedding_dim for feature in user_feature_columns])
        self.User_DNN = nn.Sequential(
            nn.Dropout(dnn_dropout),
            nn.Linear(user_DNN_input_dim, user_dnn_hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dnn_dropout),
            nn.Linear(user_dnn_hidden_units[0], user_dnn_hidden_units[1]),
        )

        item_DNN_input_dim = sum([feature.embedding_dim if isinstance(feature, SparseFeat) else feature.maxlen*feature.embedding_dim for feature in item_feature_columns])
        self.Item_DNN = nn.Sequential(
            nn.Dropout(dnn_dropout),
            nn.Linear(item_DNN_input_dim, item_dnn_hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dnn_dropout),
            nn.Linear(item_dnn_hidden_units[0], item_dnn_hidden_units[1]),
        )
        #TODO L2正则化,L2正则化可以被weight_decay所代替。
    
    def user_mebedding(self, X_user):
        user_emb = []
        length = X_user[0].size(0)
        for idx, temp in enumerate(self.user_feature_columns):
            if isinstance(temp, SparseFeat):
                user_emb.append(self.embedding_dict[temp.name].weight[X_user[idx]]) # 128 * dim
            else:
                name = temp.name.lstrip('hist_')
                user_hist_emb = [self.embedding_dict[name].weight[hist_list_idx.long().unsqueeze(0)] for hist_list_idx in X_user[idx]]
                user_hist_emb = torch.cat(user_hist_emb, dim=0).reshape(length, -1)
                user_emb.append(user_hist_emb)
        user_emb = torch.cat(user_emb, dim=1)
        return self.User_DNN(user_emb)
    
    def item_embedding(self, X_item):
        ## item
        item_emb = [self.embedding_dict[temp.name].weight[X_item[idx]] for idx, temp in enumerate(self.item_feature_columns)]
        item_emb = torch.cat(item_emb, dim=1)
        return self.Item_DNN(item_emb)
    
    def forward(self, X_user, X_item):
        # 输入的x的模式是 (batch, user_feature), (batch, item_feature)
        # 将对应的部分转化成未对应的embedding
        ##? 但是从这个意义上，hist_movieid 和 hist_genres 应该使用item的DNN来进行提取对应的，但是在这里，我们直接放在userDMM中当做历史信息提取。
        ##? 这部分计算量可能会比较大，导致模型运行速度比较小。

        #使用对应的DNN进行语义提取
        user_out_emb = self.user_mebedding(X_user)
        item_out_emb = self.item_embedding(X_item)

        #做内积，使用SIGMOD转换成点击概率
        return torch.sigmoid((user_out_emb * item_out_emb).sum(dim=1))
    
    
    


