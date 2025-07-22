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

def attSpectrumRunmodel(trainfilename,testfilename):
    X_test, y_test, name_test = read_our(testfilename)
    X_train, y_train, name_test1 = read_our(trainfilename)
    clf = xgb.XGBClassifier(scale_pos_weight = 0.220,random_state = 0, n_jobs = -1)
    clf.fit(X_train, y_train)

    # Make prediction and generate classification report
    y_pred, scores = clf.predict(X_test)
    #for i in range(len(y_pred)):
    #    if scores[i]<threshold:
    #        y_pred[i] = 0
    return name_test, scores

