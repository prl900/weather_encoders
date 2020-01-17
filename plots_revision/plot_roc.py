import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns

def get_plot(data, lmbda, mu, reg_loss):
    print('{}_{}{}_1'.format(reg_loss.lower(), lmbda, mu))
    data = data.loc[df['Model'] == '{}_{}{}_1'.format(reg_loss, lmbda, mu)]

    print(data.head())

    pod = [1] + list(data.T.iloc[1:7,0].values)
    pofd = [1] + list(data.T.iloc[7:13,0].values)

    print(pod)
    print(pofd)
    return

    tpr = pod
    fpr = pofd

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='(AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('Probability Of False Detection (POFD)')
    plt.ylabel('Probability Of Detection (POD)')
    plt.title(u"ROC {} λ={:d}, μ={:d}".format(reg_loss, lmbda, mu))
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig('roc_{}_{}{}.png'.format(reg_loss.lower(), lmbda, mu))
    print("done")

df = pd.read_csv("results_rev1.csv", header=None, names=["Model", "POD 0.2","POD 0.5","POD 1","POD 2","POD 4","POD 8", "POFD 0.2","POFD 0.5","POFD 1","POFD 2","POFD 4","POFD 8", "MAE", "MSE"])
print(df.head())
print(df["Model"])

"""
get_plot(df, 0, 0, "MAE")
get_plot(df, 1, 0, "MAE")
get_plot(df, 2, 0, "MAE")
get_plot(df, 4, 0, "MAE")
get_plot(df, 8, 0, "MAE")
get_plot(df, 0, 1, "MAE")
get_plot(df, 0, 2, "MAE")
get_plot(df, 0, 4, "MAE")
get_plot(df, 0, 8, "MAE")
get_plot(df, 1, 1, "MAE")
get_plot(df, 2, 2, "MAE")
get_plot(df, 4, 4, "MAE")
get_plot(df, 8, 8, "MAE")
"""

get_plot(df, 0, 0, "MSE")
get_plot(df, 1, 0, "MSE")
get_plot(df, 2, 0, "MSE")

"""
get_plot(df, 4, 0, "MSE")
get_plot(df, 8, 0, "MSE")
get_plot(df, 0, 1, "MSE")
get_plot(df, 0, 2, "MSE")
get_plot(df, 0, 4, "MSE")
get_plot(df, 0, 8, "MSE")
get_plot(df, 1, 1, "MSE")
get_plot(df, 2, 2, "MSE")
get_plot(df, 4, 4, "MSE")
get_plot(df, 8, 8, "MSE")
"""

