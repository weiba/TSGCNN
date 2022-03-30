# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd
import torch
from path import dir_path
sys.path.append(dir_path(k=3))
from GDSC.sampler import NewSampler
from GDSC.model import DoubleSpaceRelationGraphConvolution, Optimizer
from GDSC.utils import jaccard_coef, calculate_gene_exponent_similarity, to_data_frame


dtype = torch.float32
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_path = dir_path(k=2) + "processed_data/"

# adjcaency matrix
print("Reading cell drug...")
cell_drug = pd.read_csv(data_path + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_sum = np.sum(cell_drug, axis=1)

# cell feature
print("Reading gene...")
gene = pd.read_csv(data_path + "cell_gene.csv", index_col=0, header=0)
gene = torch.from_numpy(np.array(gene)).to(dtype=dtype, device=device)

print("Calculate feature......")
cell_sim = calculate_gene_exponent_similarity(x=gene, mu=3)

# drug feature
drug_feature = pd.read_csv(data_path + "drug_feature.csv", index_col=0, header=0)
drug_feature = torch.from_numpy(np.array(drug_feature)).to(dtype=dtype, device=device)
drug_sim = jaccard_coef(tensor=drug_feature)

# null mask
null_mask = pd.read_csv(data_path + "null_mask.csv", index_col=0, header=0)
null_mask = np.array(null_mask, dtype=np.float32)
print("Training.......")

n_kfold = 20
dim = 0
for target_index in np.arange(cell_drug.shape[dim]):
    if cell_sum[target_index] < 10:
        continue
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()
    sampler = NewSampler(adj=cell_drug, null_mask=null_mask, target_index=target_index, target_dim=dim, repeat=n_kfold)
    for train_data, test_data, train_mask, test_mask in sampler(dtype=dtype, device=device):
        model = DoubleSpaceRelationGraphConvolution(adj=train_data, x_sim=cell_sim, y_sim=drug_sim, mask=train_mask,
                                                    embed_dim=224, kernel_dim=192, alpha=6.9,
                                                    x_knn=7, y_knn=7).to(device)
        opt = Optimizer(model=model, epochs=3000, lr=5e-4, test_data=test_data, test_mask=test_mask)
        true_data, predict_data = opt()
        true_data_s = true_data_s.append(to_data_frame(data=true_data))
        predict_data_s = predict_data_s.append(to_data_frame(data=predict_data))
    true_data_s.to_csv("./result_data/cell_" + str(target_index) + "_true_data.csv")
    predict_data_s.to_csv("./result_data/cell_" + str(target_index) + "_predict_data.csv")
