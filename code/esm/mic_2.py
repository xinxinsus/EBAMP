#https://github.com/facebookresearch/esm/blob/main/examples/sup_variant_prediction.ipynb

import torch
import numpy as np
import pandas as pd

import esm
from sklearn.decomposition import PCA
FASTA_PATH = "../../attributes/predict_50000.fasta" # Path to P62593.fasta
EMB_PATH = "../../attributes/predict/" # Path to directory of embeddings for P62593.fasta
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
    cate.append(-1)

Xs = torch.stack(Xs, dim=0).numpy()

ys = np.array(ys).reshape((-1,1))
cate = np.array(cate).reshape((-1,1))

f = open('esm_mic_idxes.txt','r')
idxes = f.readlines()
for i in range(len(idxes)):
    idxes[i] = int(idxes[i].replace('\n',''))
tmp = Xs[:,idxes]

surface = np.hstack((ys,tmp,cate))
data = pd.DataFrame(surface)
writer = pd.ExcelWriter('./esm_mic_test.xlsx')
data.to_excel(writer, 'res',float_format='%.3f')
writer.save()
writer.close()
