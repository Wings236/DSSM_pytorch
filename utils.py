import torch
import numpy as np
from torch import cuda
from collections import namedtuple
DEFAULT_GROUP_NAME = "default_group"

# ======================= data class =======================
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()

# ======================= data processing =======================
import numpy as np
def gen_data_set(data, seq_max_len=50, negsample=0):
    data.sort_values("timestamp", inplace=True) # 进行时间上的排序
    item_ids = data['movie_id'].unique()
    item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values)) 
    train_set = []
    test_set = []
    for reviewerID, hist in data.groupby('user_id'):
        # 每一个按照user_id进行拆分，然后得到每一个user_id对应的movie_id，genres和rating的列表。
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()

        # 针对这个pos_list做一个负采样
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * (negsample+1), replace=True)
        
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)   # 看当前的列表长度，#? 但是实际后面不会用到

            if i != len(pos_list) - 1:
                # pos:user_id，pos_id，标签1or0，对hist倒着排，然后取前面[0,i-1]个，列表长度，流派同理，然后是当前的特征genres和rating
                train_set.append((
                    reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                    genres_list[i],
                    rating_list[i]))
                # neg:user_id, neg_id,标签0，然后把之前的都放进来，就是之前的序列这样负采样不行
                for negi in range(negsample):
                    train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
                                      genres_hist[::-1][:seq_len], item_id_genres_map[neg_list[i * negsample + negi]]))
            else:
                # test:user_id, 最后一个itemid, 1, 其他一致
                test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i], rating_list[i]))
                # 整个负的来进行对比
                test_set.append((reviewerID, neg_list[i * negsample + negi+1], 0, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i], rating_list[i]))

    
    print(f'train data size is {len(train_set)}, test data size is {len(test_set)}')
    return train_set, test_set

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
    seq_length = len(sequences)
    if maxlen is None:
        maxlen = 0
        for length in sequences:
            maxlen = max(maxlen, length)
    res = np.ones((seq_length, maxlen), dtype=dtype) * value
    for idx, seq in enumerate(sequences):
        length = len(seq)
        if length > maxlen:
            if truncating == 'pre':
                res[idx] = seq[:maxlen]
            elif truncating == 'post':
                res[idx] = seq[-maxlen:]
            else:
                raise ValueError('truncating error')
        else:
            if padding == 'pre':
                seq = [value for _ in range(maxlen-length)] + seq
            elif padding == 'post':
                seq = seq + [value for _ in range(maxlen-length)]
            else:
                raise ValueError('padding error')
            res[idx] = np.array(seq)
    return res

def gen_model_input(train_set, user_profile, seq_max_len):
    # 对数据集的信息进行拆解
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_y = np.array([line[2] for line in train_set])
    train_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_genres = [line[5] for line in train_set]
    train_genres = np.array([line[6] for line in train_set])
    
    # 让相关的信息序列长度保持一致
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_seq_genres_pad = pad_sequences(train_seq_genres, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    
    # 保存数据处理的信息
    train_X = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                        "hist_genres": train_seq_genres_pad, "hist_len": train_hist_len, "genres": train_genres}
    
    # 再增加用户本身的属性，所以前者都是交互的信息
    for key in ["gender", "age", "occupation", "zip"]:
        train_X[key] = user_profile.loc[train_X['user_id']][key].values
    
    # 输出就是X和y
    return train_X, train_y

# ======================= metrice =======================
def Precision(y_hat, y, probability=0.5):
    pred_y = y_hat > probability
    return (pred_y == y).sum()

# ======================= seed =======================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)




