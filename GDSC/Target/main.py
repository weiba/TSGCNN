# coding: utf-8
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from path import dir_path
sys.path.append(dir_path(k=3))
from GDSC.sampler import TargetSampler
from GDSC.model import DoubleSpaceRelationGraphConvolution, Optimizer


dtype = torch.float32
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_path = dir_path(k=2) + "processed_data/"
target_drug_cids = np.array([5330286, 11338033, 24825971], dtype=np.int32)

# 加载细胞系-药物矩阵
print("Reading cell drug binary ......")
cell_drug = pd.read_csv(data_path + "cell_drug_binary.csv", index_col=0, header=0)
drug_cids = cell_drug.columns.values.astype(np.int32)
cell_drug = np.array(cell_drug, dtype=np.float32)

target_indexes = common_data_index(drug_cids, target_drug_cids)

# 加载药物-指纹特征矩阵
print("Reading drug feature.....")
drug_feature = pd.read_csv(data_path + "drug_feature.csv", index_col=0, header=0)
drug_feature = torch.from_numpy(np.array(drug_feature)).to(dtype=dtype, device=device)
drug_sim = jaccard_coef(tensor=drug_feature)

# 加载细胞系-基因特征矩阵
print("Reading gene ......")
gene = pd.read_csv(data_path + "cell_gene.csv", index_col=0, header=0)
gene = torch.from_numpy(np.array(gene)).to(dtype=dtype, device=device)
cell_sim = calculate_gene_exponent_similarity(x=gene, mu=3)

# 加载null_mask
print("Reading null mask ......")
null_mask = pd.read_csv(data_path + "null_mask.csv", index_col=0, header=0)
null_mask = np.array(null_mask, dtype=np.float32)

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

n_kfolds = 5
times = []
aucs = []
for fold in range(n_kfolds):
    start = time.time()
    sampler = TargetSampler(adj=cell_drug, null_mask=null_mask, indexes=target_indexes)
    for train_data, test_data, train_mask, test_mask in sampler(dtype=dtype, device=device):
        model = DoubleSpaceRelationGraphConvolution(adj=train_data, x_sim=cell_sim, y_sim=drug_sim, mask=train_mask,
                                                    embed_dim=224, kernel_dim=192, alpha=6.9,
                                                    x_knn=7, y_knn=7).to(device)
        opt = Optimizer(model=model, epochs=3000, lr=5e-4, test_data=test_data, test_mask=test_mask)
        true_data, predict_data = opt()
        true_datas = true_datas.append(to_data_frame(data=true_data))
        predict_datas = predict_datas.append(to_data_frame(data=predict_data))
        aucs.append(roc_auc(true_data=true_data, predict_data=predict_data))
    end = time.time()
    times.append(end - start)
print("Times:", np.mean(times))
print("AUCS:", np.mean(aucs))
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")
