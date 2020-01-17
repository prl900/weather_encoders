import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns

def get_auc(data, coef, lmbda, mu, reg_loss):
    data = data.loc[df['Model'] == '{}{}_{}{}'.format(reg_loss.lower(), coef, lmbda, mu)]

    pod = [1] + list(data.T.iloc[1:7,0].values)
    pofd = [1] + list(data.T.iloc[7:13,0].values)

    tpr = pod
    fpr = pofd

    return auc(fpr, tpr)

df = pd.read_csv("results_comp.csv", header=None, names=["Model", "POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8", "MAE", "MSE"])

for coef in [0,5,2]:
    print(coef)
    print("00", get_auc(df, coef, 0, 0, "MSE"))
    print("50", get_auc(df, coef, 5, 0, "MSE"))
    print("10", get_auc(df, coef, 1, 0, "MSE"))
    print("20", get_auc(df, coef, 2, 0, "MSE"))
    print("40", get_auc(df, coef, 4, 0, "MSE"))
    print("80", get_auc(df, coef, 8, 0, "MSE"))
    print("05", get_auc(df, coef, 0, 5, "MSE"))
    print("01", get_auc(df, coef, 0, 1, "MSE"))
    print("02", get_auc(df, coef, 0, 2, "MSE"))
    print("04", get_auc(df, coef, 0, 4, "MSE"))
    print("08", get_auc(df, coef, 0, 8, "MSE"))
    print("55", get_auc(df, coef, 5, 5, "MSE"))
    print("11", get_auc(df, coef, 1, 1, "MSE"))
    print("22", get_auc(df, coef, 2, 2, "MSE"))
    print("44", get_auc(df, coef, 4, 4, "MSE"))
    print("88", get_auc(df, coef, 8, 8, "MSE"))
