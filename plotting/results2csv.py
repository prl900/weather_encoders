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
    #hits = np.nansum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    #misses = np.nansum(np.logical_and((y_obs > threshold), (y_pred < threshold)))
    hits = np.nansum(np.logical_and((y_obs > 1.), (y_pred > threshold)))
    misses = np.nansum(np.logical_and((y_obs > 1.), (y_pred < threshold)))

    return hits/(hits+misses)


def verif_pofd(y_obs, y_pred, threshold):
    # Probability of false detections = False Alarms / (False Alarms + True Negatives)
    #true_neg = np.nansum(np.logical_and((y_obs < threshold), (y_pred < threshold)))
    #f_alarms = np.nansum(np.logical_and((y_obs < threshold), (y_pred > threshold)))
    true_neg = np.nansum(np.logical_and((y_obs < 1.), (y_pred < threshold)))
    f_alarms = np.nansum(np.logical_and((y_obs < 1.), (y_pred > threshold)))
    
    return f_alarms/(f_alarms+true_neg)

def verif_mse(y_obs, y_pred):
    
    return np.mean(np.square(y_obs-y_pred))

def verif_mae(y_obs, y_pred):
    
    return np.mean(np.abs(y_obs-y_pred))

def get_stats(modelh5, lmbda, mu, x_test, y_test):
    print(modelh5[13:19], end =",")

    model = load_model(modelh5, custom_objects={'comb_mse': closs.get_diff_comb_mse_loss(1., lmbda, mu), 'comb_mae': closs.get_diff_comb_mae_loss(1., lmbda, mu), 'pod': closs.get_pod_loss(1.), 'pom': closs.get_pom_loss(1.), 'far': closs.get_far_loss(1.), 'pofd': closs.get_pofd_loss(1.)})

    y_pred = model.predict(x_test)

    for v in [.2,.5,1.,2.,5.,10.]:
        pod = verif_pod(y_test, y_pred, v)
        print(pod, end =",")
    for v in [.2,.5,1.,2.,5.,10.]:
        pofd = verif_pofd(y_test, y_pred, v)
        print(pofd, end =",")

    print(verif_mae(y_test, y_pred), end =",")
    print(verif_mse(y_test, y_pred))

x = np.load("/data/ERA-Int/10zlevels_min.npy")
y = np.log(1+np.load("/data/ERA-Int/tp_min.npy"))

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :, :]
x_test = x[14000:, :]
x = None

y = y[idxs, :, :, None]
y_test = y[14000:, :]
y = None

"""
print("MAE:")
get_stats('../unet_comb_mae_00_10levels.h5', .0, .0, x_test, y_test)

get_stats('../unet_comb_mae_10_10levels.h5', 1, .0, x_test, y_test)
get_stats('../unet_comb_mae_20_10levels.h5', 2, .0, x_test, y_test)
get_stats('../unet_comb_mae_40_10levels.h5', 4, .0, x_test, y_test)
get_stats('../unet_comb_mae_80_10levels.h5', 8, .0, x_test, y_test)

get_stats('../unet_comb_mae_01_10levels.h5', .0, 1, x_test, y_test)
get_stats('../unet_comb_mae_02_10levels.h5', .0, 2, x_test, y_test)
get_stats('../unet_comb_mae_04_10levels.h5', .0, 4, x_test, y_test)
get_stats('../unet_comb_mae_08_10levels.h5', .0, 8, x_test, y_test)

get_stats('../unet_comb_mae_11_10levels.h5', 1, 1, x_test, y_test)
get_stats('../unet_comb_mae_22_10levels.h5', 2, 2, x_test, y_test)
get_stats('../unet_comb_mae_44_10levels.h5', 4, 4, x_test, y_test)
get_stats('../unet_comb_mae_88_10levels.h5', 8, 8, x_test, y_test)
"""

print("MSE:")
get_stats('../unet_comb_mse_55_10levels.h5', .0, .0, x_test, y_test)

"""
get_stats('../unet_comb_mse_10_10levels.h5', 1, .0, x_test, y_test)
get_stats('../unet_comb_mse_20_10levels.h5', 2, .0, x_test, y_test)
get_stats('../unet_comb_mse_40_10levels.h5', 4, .0, x_test, y_test)
get_stats('../unet_comb_mse_80_10levels.h5', 8, .0, x_test, y_test)

get_stats('../unet_comb_mse_01_10levels.h5', .0, 1, x_test, y_test)
get_stats('../unet_comb_mse_02_10levels.h5', .0, 2, x_test, y_test)
get_stats('../unet_comb_mse_04_10levels.h5', .0, 4, x_test, y_test)
get_stats('../unet_comb_mse_08_10levels.h5', .0, 8, x_test, y_test)

get_stats('../unet_comb_mse_11_10levels.h5', 1, 1, x_test, y_test)
get_stats('../unet_comb_mse_22_10levels.h5', 2, 2, x_test, y_test)
get_stats('../unet_comb_mse_44_10levels.h5', 4, 4, x_test, y_test)
get_stats('../unet_comb_mse_88_10levels.h5', 8, 8, x_test, y_test)
"""
