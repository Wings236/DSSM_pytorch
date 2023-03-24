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
        total_feature = user_feature_columns + item_feature_columns
        dim = total_feature[0].embedding_dim
        total_num = sum([feat.vocabulary_size for feat in total_feature])
        self.embedding = nn.Embedding(total_num, dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0)
        
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
        user_emb = self.embedding.weight[X_user].reshape(X_user.shape[0], -1)
        return self.User_DNN(user_emb)
    
    def item_embedding(self, X_item):
        item_emb = self.embedding.weight[X_item].reshape(X_item.shape[0], -1)
        return self.Item_DNN(item_emb)
    
    def forward(self, X_user, X_item):
        # 输入的x的格式是 (batch, user_feature), (batch, item_feature)
        #使用对应的DNN进行语义提取
        user_out_emb = self.user_mebedding(X_user)
        item_out_emb = self.item_embedding(X_item)

        #做内积，使用SIGMOD转换成点击概率
        return torch.sigmoid((user_out_emb * item_out_emb).sum(dim=1))
    
    
    


