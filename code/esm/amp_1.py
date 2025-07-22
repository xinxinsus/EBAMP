#https://github.com/facebookresearch/esm/blob/main/examples/sup_variant_prediction.ipynb


import random
from collections import Counter
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import esm
import scipy
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import Pipeline




FASTA_PATH = "../../attributes/amp_p.fasta" # Path to P62593.fasta
EMB_PATH = "../../attributes/amp_p/" # Path to directory of embeddings for P62593.fasta
EMB_LAYER = 33
ys = []
Xs = []
cate = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('|')[-1]
    ys.append(scaled_effect)
    fn = f'{EMB_PATH}/{str(scaled_effect)}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
    cate.append(1)
FASTA_PATH = "../../attributes/amp_n.fasta" # Path to P62593.fasta
EMB_PATH = "../../attributes/amp_n/" # Path to directory of embeddings for P62593.fasta
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('|')[-1]
    ys.append(scaled_effect)
    fn = f'{EMB_PATH}/{str(scaled_effect)}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
    cate.append(0)

Xs = torch.stack(Xs, dim=0).numpy()

ys = np.array(ys).reshape((-1,1))
cate = np.array(cate).reshape((-1,1))

num_pca_components = 0.95
pca = PCA(num_pca_components)
Xs_pca = pca.fit_transform(Xs)

k1_spss = pca.components_.T
weight = (np.dot(k1_spss,pca.explained_variance_ratio_))/np.sum(pca.explained_variance_ratio_)
weighted_weight = weight/np.sum(weight)
max_location = sorted(enumerate(weighted_weight),key=lambda y:y[1],reverse=True)

f = open('./esm_amp_idxes.txt','w')
idxes = []
for i in range(len(Xs_pca[0])):
    idx = max_location[i]
    idxes.append(idx[0])
    f.write(str(idx[0])+'\n')
f.close()

tmp = Xs[:,idxes]
surface = np.hstack((ys,tmp,cate))
data = pd.DataFrame(surface)
writer = pd.ExcelWriter('./esm_amp_train.xlsx')
data.to_excel(writer, 'res',float_format='%.3f')
writer.save()
writer.close()