import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold
import torch


class RandomSampler(object):
    def __init__(self, adj: np.ndarray, null_mask: np.ndarray, cv_num=5, shuffle=True, random_state=21):
        super(RandomSampler, self).__init__()
        self.adj = adj
        self.null_mask = null_mask
        self.kfold = KFold(n_splits=cv_num, shuffle=shuffle, random_state=random_state)

    def sample_train_test_data(self, train_index: np.ndarray, test_index: np.ndarray):
        sp_adj = sp.coo_matrix(self.adj)
        train_data = sp.coo_matrix((sp_adj.data[train_index], (sp_adj.row[train_index], sp_adj.col[train_index])),
                                   shape=self.adj.shape).toarray()
        test_data = sp.coo_matrix((sp_adj.data[test_index], (sp_adj.row[test_index], sp_adj.col[test_index])),
                                  shape=self.adj.shape).toarray()
        return train_data, test_data

    def sampler_train_test_mask(self, test_index: np.ndarray):
        neg_value = np.ones_like(self.adj) - self.adj - self.null_mask
        sp_neg_value = sp.coo_matrix(neg_value)
        total_index = np.arange(sp_neg_value.data.shape[0])
        neg_test_index = np.random.choice(total_index, test_index.shape[0], replace=False)
        neg_test_mask = sp.coo_matrix((sp_neg_value.data[neg_test_index], (sp_neg_value.row[neg_test_index],
                                                                           sp_neg_value.col[neg_test_index])),
                                      shape=self.adj.shape).toarray()
        neg_train_mask = neg_value - neg_test_mask
        return neg_train_mask, neg_test_mask

    def __call__(self, dtype, device):
        total_number = sp.coo_matrix(self.adj).data.shape[0]
        for train_index, test_index in self.kfold.split(np.arange(total_number)):
            train_data, test_data = self.sample_train_test_data(train_index=train_index, test_index=test_index)
            train_mask, test_mask = self.sampler_train_test_mask(test_index=test_index)
            train_mask = (train_data + train_mask).astype(np.bool)
            test_mask = (test_data + test_mask).astype(np.bool)
            train_data = torch.from_numpy(train_data).to(dtype=dtype, device=device)
            test_data = torch.from_numpy(test_data).to(dtype=dtype, device=device)
            train_mask = torch.from_numpy(train_mask).to(dtype=torch.bool, device=device)
            test_mask = torch.from_numpy(test_mask).to(dtype=torch.bool, device=device)
            yield train_data, test_data, train_mask, test_mask


class RandomSamplerOpposite(object):
    def __init__(self, adj: np.ndarray, null_mask: np.ndarray, cv_num=5, shuffle=True, random_state=21):
        super(RandomSamplerOpposite, self).__init__()
        self.adj = adj
        self.null_mask = null_mask
        self.kfold = KFold(n_splits=cv_num, shuffle=shuffle, random_state=random_state)

    def sample_train_test_data(self, train_index: np.ndarray, test_index: np.ndarray):
        sp_adj = sp.coo_matrix(self.adj)
        train_data = sp.coo_matrix((sp_adj.data[train_index], (sp_adj.row[train_index], sp_adj.col[train_index])),
                                   shape=self.adj.shape).toarray()
        test_data = sp.coo_matrix((sp_adj.data[test_index], (sp_adj.row[test_index], sp_adj.col[test_index])),
                                  shape=self.adj.shape).toarray()
        return train_data, test_data

    def sampler_train_test_mask(self, train_index: np.ndarray):
        neg_value = np.ones_like(self.adj) - self.adj - self.null_mask
        sp_neg_value = sp.coo_matrix(neg_value)
        total_index = np.arange(sp_neg_value.data.shape[0])
        neg_train_index = np.random.choice(total_index, train_index.shape[0], replace=False)
        neg_train_mask = sp.coo_matrix((sp_neg_value.data[neg_train_index], (sp_neg_value.row[neg_train_index],
                                                                             sp_neg_value.col[neg_train_index])),
                                       shape=self.adj.shape).toarray()
        neg_test_mask = neg_value - neg_train_mask
        return neg_train_mask, neg_test_mask

    def __call__(self, dtype, device):
        total_number = sp.coo_matrix(self.adj).data.shape[0]
        for train_index, test_index in self.kfold.split(np.arange(total_number)):
            train_data, test_data = self.sample_train_test_data(train_index=train_index, test_index=test_index)
            train_mask, test_mask = self.sampler_train_test_mask(train_index=train_index)
            train_mask = (train_data + train_mask).astype(np.bool)
            test_mask = (test_data + test_mask).astype(np.bool)
            train_data = torch.from_numpy(train_data).to(dtype=dtype, device=device)
            test_data = torch.from_numpy(test_data).to(dtype=dtype, device=device)
            train_mask = torch.from_numpy(train_mask).to(dtype=torch.bool, device=device)
            test_mask = torch.from_numpy(test_mask).to(dtype=torch.bool, device=device)
            yield train_data, test_data, train_mask, test_mask


class NewSampler(object):
    def __init__(self, adj: np.ndarray, null_mask: np.ndarray, target_index: int, target_dim: 0 or 1, repeat=20):
        super(NewSampler, self).__init__()
        print("That is OK!")
        self.adj = adj
        self.null_mask = null_mask
        self.dim = target_dim
        self.target_index = target_index
        self.repeat = repeat

    def sample_target_test_index(self):
        if self.dim:
            target_pos_index = np.where(self.adj[:, self.target_index] == 1)[0]
        else:
            target_pos_index = np.where(self.adj[self.target_index, :] == 1)[0]
        return target_pos_index

    def sample_train_test_data(self):
        test_data = np.zeros(self.adj.shape, dtype=np.float32)
        test_index = self.sample_target_test_index()
        if self.dim:
            test_data[test_index, self.target_index] = 1
        else:
            test_data[self.target_index, test_index] = 1
        train_data = self.adj - test_data
        return train_data, test_data

    def sample_train_test_mask(self):
        test_index = self.sample_target_test_index()
        neg_value = np.ones(self.adj.shape, dtype=np.float32)
        neg_value = neg_value - self.adj - self.null_mask
        neg_test_mask = np.zeros(self.adj.shape, dtype=np.float32)
        if self.dim:
            target_neg_index = np.where(neg_value[:, self.target_index] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(target_neg_index, test_index.shape[0], replace=False)
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[target_neg_test_index, self.target_index] = 1
            neg_value[:, self.target_index] = 0
        else:
            target_neg_index = np.where(neg_value[self.target_index, :] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(target_neg_index, test_index.shape[0], replace=False)
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[self.target_index, target_neg_test_index] = 1
            neg_value[self.target_index, :] = 0
        return neg_value, neg_test_mask

    def __call__(self, dtype, device):
        for i in range(self.repeat):
            train_data, test_data = self.sample_train_test_data()
            train_mask, test_mask = self.sample_train_test_mask()
            train_mask = (train_data + train_mask).astype(np.bool)
            test_mask = (test_data + test_mask).astype(np.bool)
            train_data = torch.from_numpy(train_data).to(dtype=dtype, device=device)
            test_data = torch.from_numpy(test_data).to(dtype=dtype, device=device)
            train_mask = torch.from_numpy(train_mask).to(dtype=torch.bool, device=device)
            test_mask = torch.from_numpy(test_mask).to(dtype=torch.bool, device=device)
            yield train_data, test_data, train_mask, test_mask


class SingleSampler(object):
    def __init__(self, adj: np.ndarray, null_mask: np.ndarray, index: int, cv_num=5, shuffle=True, random_state=21):
        super(SingleSampler, self).__init__()
        self.adj = adj
        self.null_mask = null_mask
        self.index = index
        self.kfold = KFold(n_splits=cv_num, shuffle=shuffle, random_state=random_state)

    def sample_train_test_data(self, test_index: np.ndarray):
        test_data = np.zeros(self.adj.shape, dtype=np.float32)
        test_data[test_index, self.index] = 1
        train_data = self.adj - test_data
        return train_data, test_data

    def sample_train_test_neg_mask(self, test_index: np.ndarray):
        neg_value = np.ones(self.adj.shape, dtype=np.float32)
        neg_value = neg_value - self.adj - self.null_mask
        neg_test_mask = np.zeros(self.adj.shape, dtype=np.float32)
        target_neg_index = np.where(neg_value[:, self.index] == 1)[0]
        target_neg_test_index = np.random.choice(target_neg_index, test_index.shape[0], replace=False)
        neg_test_mask[target_neg_test_index, self.index] = 1
        neg_value[target_neg_test_index, self.index] = 0
        return neg_value, neg_test_mask

    def __call__(self, dtype, device):
        target_pos_index = np.where(self.adj[:, self.index] == 1)[0]
        n = target_pos_index.shape[0]
        for _, test_id in self.kfold.split(range(n)):
            test_index = target_pos_index[test_id]
            train_data, test_data = self.sample_train_test_data(test_index=test_index)
            train_mask, test_mask = self.sample_train_test_neg_mask(test_index=test_index)
            train_mask = (train_data + train_mask).astype(np.bool)
            test_mask = (test_data + test_mask).astype(np.bool)
            train_data = torch.from_numpy(train_data).to(dtype=dtype, device=device)
            test_data = torch.from_numpy(test_data).to(dtype=dtype, device=device)
            train_mask = torch.from_numpy(train_mask).to(dtype=torch.bool, device=device)
            test_mask = torch.from_numpy(test_mask).to(dtype=torch.bool, device=device)
            yield train_data, test_data, train_mask, test_mask


class TargetSampler(object):
    def __init__(self, adj: np.ndarray, null_mask: np.ndarray, indexes: np.ndarray, cv_num=5, shuffle=True,
                 random_state=21):
        super(TargetSampler, self).__init__()
        self.adj = adj
        self.null_mask = null_mask
        self.indexes = indexes
        self.kfold = KFold(n_splits=cv_num, shuffle=shuffle, random_state=random_state)

    def sample_train_test_data(self, pos_train_index: np.ndarray, pos_test_index: np.ndarray):
        n_target = self.indexes.shape[0]
        target_response = self.adj[:, self.indexes].reshape((-1, n_target))
        train_data = self.adj.copy()
        train_data[:, self.indexes] = 0
        target_pos_value = sp.coo_matrix(target_response)
        target_train_data = sp.coo_matrix((target_pos_value.data[pos_train_index],
                                           (target_pos_value.row[pos_train_index],
                                            target_pos_value.col[pos_train_index])),
                                          shape=target_response.shape).toarray()
        target_test_data = sp.coo_matrix((target_pos_value.data[pos_test_index],
                                          (target_pos_value.row[pos_test_index],
                                           target_pos_value.col[pos_test_index])),
                                         shape=target_response.shape).toarray()
        test_data = np.zeros(self.adj.shape, dtype=np.float32)
        for i, value in enumerate(self.indexes):
            train_data[:, value] = target_train_data[:, i]
            test_data[:, value] = target_test_data[:, i]
        return train_data, test_data

    def sample_train_test_mask(self, test_number: int):
        neg_value = np.ones(self.adj.shape, dtype=np.float32) - self.adj - self.null_mask
        target_neg = neg_value[:, self.indexes].reshape((-1, self.indexes.shape[0]))
        sp_target_neg = sp.coo_matrix(target_neg)
        ids = np.arange(sp_target_neg.data.shape[0])
        target_neg_test_index = np.random.choice(ids, test_number, replace=False)
        target_neg_test_mask = sp.coo_matrix((sp_target_neg.data[target_neg_test_index],
                                              (sp_target_neg.row[target_neg_test_index],
                                               sp_target_neg.col[target_neg_test_index])),
                                             shape=target_neg.shape).toarray()
        neg_test_mask = np.zeros(self.adj.shape, dtype=np.float32)
        for i, value in enumerate(self.indexes):
            neg_test_mask[:, value] = target_neg_test_mask[:, i]
        neg_train_mask = neg_value - neg_test_mask
        return neg_train_mask, neg_test_mask

    def __call__(self, dtype, device):
        target_adj = self.adj[:, self.indexes].reshape((-1, self.indexes.shape[0]))
        sp_target_adj = sp.coo_matrix(target_adj)
        for train_index, test_index in self.kfold.split(range(sp_target_adj.data.shape[0])):
            test_index = np.array(test_index)
            train_data, test_data = self.sample_train_test_data(pos_train_index=train_index, pos_test_index=test_index)
            train_mask, test_mask = self.sample_train_test_mask(test_number=test_index.shape[0])
            train_mask = (train_data + train_mask).astype(np.bool)
            test_mask = (test_mask + test_data).astype(np.bool)
            train_data = torch.from_numpy(train_data).to(dtype=dtype, device=device)
            test_data = torch.from_numpy(test_data).to(dtype=dtype, device=device)
            train_mask = torch.from_numpy(train_mask).to(dtype=torch.bool, device=device)
            test_mask = torch.from_numpy(test_mask).to(dtype=torch.bool, device=device)
            yield train_data, test_data, train_mask, test_mask
