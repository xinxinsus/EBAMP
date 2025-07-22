"""
=============================
LCEClassifier on Iris dataset
=============================

An example of :class:`lce.LCEClassifier`
"""
import pandas as pd
import numpy as np

import xgboost as xgb
def read_our(filename):
    sheet = pd.read_excel(filename, 'res')
    validList=sheet.values
    tmp = len(validList[0]) - 1
    y_valid = pd.DataFrame(sheet, columns=[tmp-1])
    names = pd.DataFrame(sheet, columns=[0])

    valid_pdbs=[]
    for item in validList:
        valid_pdbs.append(item)
    valid_pdbs = np.array(valid_pdbs)

    da=valid_pdbs[0:,2:tmp]
    data=[]
    for i in da:
        temp=[]
        for j in i:
            temp.append(float(j))
        data.append(temp)
    data = np.array(data).tolist()
    y_valid = np.array(y_valid).tolist()
    names = np.array(names).tolist()
    returnNames = []
    for idx in range(len(names)):
        returnNames.append(names[idx][0])
    return data, y_valid, np.array(returnNames)

def attMicRunmodel(trainfilename,testfilename):
    X_test, y_test, name_test = read_our(testfilename)
    X_train, y_train, name_test1 = read_our(trainfilename)
    clf = xgb.XGBClassifier(scale_pos_weight = 3.576,random_state = 0, n_jobs = -1)
    clf.fit(X_train, y_train)

    # Make prediction and generate classification report
    y_pred, scores = clf.predict(X_test)
    #for i in range(len(y_pred)):
    #    if scores[i]<threshold:
    #        y_pred[i] = 0
    return name_test, scores


def attMicRunmodelReg(trainfilename,testfilename):
    X_train, y_train, name_test1 = read_our(trainfilename)

    sheet = pd.read_excel('./data/micpos.xlsx', 'res')
    validList = sheet.values
    ids = validList[:, 0]
    micvalues = validList[:, 2]
    reg_Xtrain = []
    reg_Ytrain = []
    for i in range(len(ids)):
        for j in range(len(name_test1)):
            if name_test1[j] == ids[i] and y_train[j][0] == 1:
                reg_Xtrain.append(X_train[j])
                reg_Ytrain.append(micvalues[i])
                break

    reg_Ytrain = np.array(reg_Ytrain).reshape((-1, 1))
    X_test, y_test, name_test = read_our(testfilename)

    clf = xgb.XGBRegressor(use_label_encoder=False,objective="rank:pairwise")
    clf.fit(reg_Xtrain, reg_Ytrain)
    y_pred = clf.predict(X_test)

    res = []
    idxes = np.argsort(y_pred)
    for idx in idxes:
        res.append([name_test[idx],y_pred[idx]])

    data = pd.DataFrame(res)
    writer = pd.ExcelWriter('att_pre_micvalues.xlsx')
    data.to_excel(writer, 'res', float_format='%.3f')
    writer.save()
    writer.close()

#attMicRunmodelReg('/home/dell/Documents/GPTforBio/att/att_mic_train.xlsx',
#                                                '/home/dell/Documents/GPTforBio/att/att_mic_test.xlsx')