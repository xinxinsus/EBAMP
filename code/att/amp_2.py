
import torch
import numpy as np
import pandas as pd

import esm
from sklearn.decomposition import PCA
sheet = pd.read_excel('./att_pre_raw.xlsx','res')
validList = sheet.values
np.random.shuffle(validList)
Xs = validList[:,2:len(validList[0])-1]
ys = validList[:,1]
cate = validList[:,len(validList[0])-1]

#Xs = torch.stack(Xs, dim=0).numpy()

ys = np.array(ys).reshape((-1,1))
cate = np.array(cate).reshape((-1,1))

f = open('att_amp_idxes.txt','r')
idxes = f.readlines()
for i in range(len(idxes)):
    idxes[i] = int(idxes[i].replace('\n',''))
tmp = Xs[:,idxes]

surface = np.hstack((ys,tmp,cate))
data = pd.DataFrame(surface)
writer = pd.ExcelWriter('./att_amp_test.xlsx')
data.to_excel(writer, 'res',float_format='%.3f')
writer.save()
writer.close()
