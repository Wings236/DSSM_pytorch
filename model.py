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
        :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
        :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
        :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param loss_type: string. Loss type.
        :param temperature: float. Scaling factor.
        :param sampler_config: negative sample config.
        :param seed: integer ,to use as random seed.
        :return: A Keras model instance.
        '''
        # 参数初始化
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        total_feature = user_feature_columns + item_feature_columns
        # 不管你是变长还是其他，都可以通过取对应这个位置总数来充当对应的词向量大小
        self.embedding_dict = nn.ModuleDict({feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim) for feat in total_feature})
        # print(self.user_feature_columns, self.item_feature_columns, self.embedding_dict)
        # 初始化
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.01)
        # print(user_feature_columns)
        # 然后再针对对应的位置来进行取对应的emb，具体就是锁定对应的地方能够给到多少个emb来聚合
        # 本质上就是针对每一个输入的数据，寻找对应的emb，然后聚合起来送入DNN输出一个向量，然后再进行聚合即可
        # 如果是sparse的话，就占一个位置，如果是Varlen的话，就占maxlen个位置，然后把这个给cancat起来。
        # 所以每一个DNN的输入的维度应该是dim*Sparse个数+dim*maxlen*Varlen的个数
        ## user
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
        #TODO L2正则化

    def forward(self, X_user, X_item):
        # 输入的x的模式是 (batch, user_feature), (batch, item_feature)
        # 将对应的部分转化成未对应的embedding
        # userid, movieid, hist_movieid, hist_genres, hist_len, genres, gender, age, occ, zip
        ## user, userid, hist_movieid, hist_genres, genres, age, occ, zip#? hist_len 其实是无用数据
        ## 但是从这个意义上，hist_movieid 和 hist_genres 和应该使用item的DNN来进行提取，但是在这里，我们直接无视。
        user_emb = []
        length = X_user[0].size(0)
        for idx, temp in enumerate(self.user_feature_columns):
            if isinstance(temp, SparseFeat):
                user_emb.append(self.embedding_dict[temp.name].weight[X_user[idx]]) # 128 * dim
            else:
                user_hist_emb = []
                name = temp.name.lstrip('hist_')
                for hist_list_idx in X_user[idx]:
                    user_hist_emb.append(self.embedding_dict[name].weight[hist_list_idx.long().unsqueeze(0)])
                user_hist_emb = torch.cat(user_hist_emb, dim=0).reshape(length, -1)
                user_emb.append(user_hist_emb)
        user_emb = torch.cat(user_emb, dim=1)
        
        ## item
        item_emb = []
        for idx, temp in enumerate(self.item_feature_columns):
            item_emb.append(self.embedding_dict[temp.name].weight[X_item[idx]])
        item_emb = torch.cat(item_emb, dim=1)

        #使用对应的DNN进行语义提取
        user_out_emb = self.User_DNN(user_emb)
        item_out_emb = self.Item_DNN(item_emb)

        #做内积，使用SIGMOD转换成概率
        return torch.sigmoid((user_out_emb * item_out_emb).sum(dim=1))

