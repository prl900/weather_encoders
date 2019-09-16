import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import sys
sys.path.append('../categorical_losses')

import categorical_losses as closs

def verif_pod(y_obs, y_pred, threshold):
    # Probability of detection = Hits / (Hits + Misses)
    print("POD", threshold)
    #hits = np.nansum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    #misses = np.nansum(np.logical_and((y_obs > threshold), (y_pred < threshold)))
    hits = np.nansum(np.logical_and((y_obs > 1.), (y_pred > threshold)))
    misses = np.nansum(np.logical_and((y_obs > 1.), (y_pred < threshold)))

    print(hits/(hits+misses))
    return hits/(hits+misses)


def verif_pofd(y_obs, y_pred, threshold):
    # Probability of false detections = False Alarms / (False Alarms + True Negatives)
    print("POFD", threshold)
    #true_neg = np.nansum(np.logical_and((y_obs < threshold), (y_pred < threshold)))
    #f_alarms = np.nansum(np.logical_and((y_obs < threshold), (y_pred > threshold)))
    true_neg = np.nansum(np.logical_and((y_obs < 1.), (y_pred < threshold)))
    f_alarms = np.nansum(np.logical_and((y_obs < 1.), (y_pred > threshold)))
    
    print(f_alarms/(f_alarms+true_neg))
    return f_alarms/(f_alarms+true_neg)

def get_plot(modelh5, lmbda, mu):
    model = load_model(modelh5, custom_objects={'comb_mse': closs.get_diff_comb_mse_loss(1., lmbda, mu), 'comb_mae': closs.get_diff_comb_mae_loss(1., lmbda, mu), 'pod': closs.get_pod_loss(1.), 'pom': closs.get_pom_loss(1.), 'far': closs.get_far_loss(1.), 'pofd': closs.get_pofd_loss(1.)})

    y_pred = model.predict(x_test)
    np.save(modelh5[:-3], y_pred)

    tpr = []
    tpr.append(1)
    fpr = []
    fpr.append(1)
    for v in [.2,.5,1.,2.,5.,10.]:
        tpr.append(verif_pod(y_test, y_pred, v))
        fpr.append(verif_pofd(y_test, y_pred, v))

    roc_auc = auc(fpr, tpr)
    print(tpr)
    print(fpr)
    print(roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format(modelh5[3:-3]))
    print("done")

"""
print(y_pred.shape)
print(y_test.shape)
print(verif_pod(y_test, y_pred, 0.1))
print(verif_pod(y_test, y_pred, 0.5))
print(verif_pod(y_test, y_pred, 1.))
print(verif_pod(y_test, y_pred, 2.))
print(verif_pod(y_test, y_pred, 3.))
print(verif_pod(y_test, y_pred, 4.))
print(verif_pod(y_test, y_pred, 5.))
print(10*'-')
print("Hits         3.", np.sum(np.logical_and((y_test > 3.), (y_pred > 3.))))
print("Misses       3.", np.sum(np.logical_and((y_test > 3.), (y_pred < 3.))))
print("False Alarms 3.", np.sum(np.logical_and((y_test < 3.), (y_pred > 3.))))
print(10*'-')
print("Hits         5.", np.sum(np.logical_and((y_test > 5.), (y_pred > 5.))))
print("Misses       5.", np.sum(np.logical_and((y_test > 5.), (y_pred < 5.))))
print("False Alarms 5.", np.sum(np.logical_and((y_test < 5.), (y_pred > 5.))))
print(10*'-')
print("0.1", verif_pod(y_test, y_pred, 0.1), verif_pofd(y_test, y_pred, 0.1))
print("0.5", verif_pod(y_test, y_pred, 0.5), verif_pofd(y_test, y_pred, 0.5))
print("1.0", verif_pod(y_test, y_pred, 1.), verif_pofd(y_test, y_pred, 1.))
print("2.0", verif_pod(y_test, y_pred, 2.), verif_pofd(y_test, y_pred, 2.))
print("3.0", verif_pod(y_test, y_pred, 3.), verif_pofd(y_test, y_pred, 3.))
print("4.0", verif_pod(y_test, y_pred, 4.), verif_pofd(y_test, y_pred, 4.))
print("5.0", verif_pod(y_test, y_pred, 5.), verif_pofd(y_test, y_pred, 5.))
"""

x = np.load("/data/ERA-Int/10zlevels_min.npy")
print("A")
y = np.log(1+np.load("/data/ERA-Int/tp_min.npy"))
print("B")

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :, :]
x_test = x[14000:, :]
x = None

y = y[idxs, :, :, None]
y_test = y[14000:, :]
y = None

print("MAE:")
get_plot('../unet_comb_mae_00_10levels.h5', .0, .0)

get_plot('../unet_comb_mae_10_10levels.h5', 1., .0)
get_plot('../unet_comb_mae_20_10levels.h5', 2., .0)
get_plot('../unet_comb_mae_40_10levels.h5', 4., .0)
get_plot('../unet_comb_mae_80_10levels.h5', 8., .0)

get_plot('../unet_comb_mae_01_10levels.h5', 0., 1)
get_plot('../unet_comb_mae_02_10levels.h5', 0., 2)
get_plot('../unet_comb_mae_04_10levels.h5', 0., 4)
get_plot('../unet_comb_mae_08_10levels.h5', 0., 8)

get_plot('../unet_comb_mae_11_10levels.h5', 1., 1)
get_plot('../unet_comb_mae_22_10levels.h5', 2., 2)
get_plot('../unet_comb_mae_44_10levels.h5', 4., 4)
get_plot('../unet_comb_mae_88_10levels.h5', 8., 8)

print("MSE:")
get_plot('../unet_comb_mse_00_10levels.h5', .0, .0)

get_plot('../unet_comb_mse_10_10levels.h5', 1., .0)
get_plot('../unet_comb_mse_20_10levels.h5', 2., .0)
get_plot('../unet_comb_mse_40_10levels.h5', 4., .0)
get_plot('../unet_comb_mse_80_10levels.h5', 8., .0)

get_plot('../unet_comb_mse_01_10levels.h5', 0., 1)
get_plot('../unet_comb_mse_02_10levels.h5', 0., 2)
get_plot('../unet_comb_mse_04_10levels.h5', 0., 4)
get_plot('../unet_comb_mse_08_10levels.h5', 0., 8)

get_plot('../unet_comb_mse_11_10levels.h5', 1., 1)
get_plot('../unet_comb_mse_22_10levels.h5', 2., 2)
get_plot('../unet_comb_mse_44_10levels.h5', 4., 4)
get_plot('../unet_comb_mse_88_10levels.h5', 8., 8.)
