#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.manifold import TSNE

from learn.dataset import TabularDataset
from learn.model import CiteAutoencoder
from learn.train import train_model, get_encodings
import umap


data_path = Path("./data/")
list(data_path.iterdir())

rna = pd.read_csv(data_path/"rna_scale.csv.gz", index_col=0).T
rna = rna.reset_index(drop=True)
rna.head()

pro = pd.read_csv(data_path/"protein_scale.csv.gz", index_col=0).T
pro = pro.reset_index(drop=True)
pro.head()
nfeatures_rna = rna.shape[1]
nfeatures_pro = pro.shape[1]

print(nfeatures_rna, nfeatures_pro)

# concat rna and pro
citeseq = pd.concat([rna, pro], axis=1)
citeseq.head()

train, valid = train_test_split(citeseq.to_numpy(dtype=np.float32), test_size=0.1, random_state=0)

train_ds = TabularDataset(train)
valid_ds = TabularDataset(valid)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False)

model = CiteAutoencoder(nfeatures_rna, nfeatures_pro, hidden_rna=85,
                        hidden_pro=15, z_dim=20)

lr = 1e-2
epochs = 50
model, losses = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs)

test_ds = TabularDataset(citeseq.to_numpy(dtype=np.float32))
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

encodings = get_encodings(model, test_dl)
encodings = encodings.cpu().numpy()

metadata = pd.read_csv(data_path/"metadata.csv.gz", index_col=0)

metadata["celltype.l1.5"] = metadata["celltype.l1"].values
metadata.loc[metadata["celltype.l2"].str.startswith("CD4"), "celltype.l1.5"] = "CD4 T"
metadata.loc[metadata["celltype.l2"].str.startswith("CD8"), "celltype.l1.5"] = "CD8 T"
metadata.loc[metadata["celltype.l2"]=="Treg", "celltype.l1.5"] = "CD4 T"
metadata.loc[metadata["celltype.l2"]=="MAIT", "celltype.l1.5"] = "MAIT"
metadata.loc[metadata["celltype.l2"]=="gdT", "celltype.l1.5"] = "gdT"

X_tsne = TSNE(n_components=2,random_state=0).fit_transform(encodings)


plot_dat=metadata.copy()
plot_dat["UMAP1"]=X_tsne[:,0]
plot_dat["UMAP2"]=X_tsne[:,1]

plot_dat.to_csv('tsneRes.csv',mode='a')
