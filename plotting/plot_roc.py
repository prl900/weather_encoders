import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from ../categorical_losses import categorical_losses as closs

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

model = load_model('unet_mae_10levels.h5', custom_objects={'pod': closs.get_diff_pod_loss(.5), 'far': closs.get_diff_far_loss(.5)})

def verif_pod(y_obs, y_pred, threshold):
    # Probability of detection = Hits / (Hits + Misses)
    hits = np.sum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    misses = np.sum(np.logical_and((y_obs > threshold), (y_pred < threshold)))

    return hits/(hits+misses)

def verif_pofd(y_obs, y_pred, threshold):
    # Probability of false detections = False Alarms / (False Alarms + True Negatives)
    true_neg = np.sum(np.logical_and((y_obs < threshold), (y_pred < threshold)))
    f_alarms = np.sum(np.logical_and((y_obs < threshold), (y_pred > threshold)))
    
    return f_alarms/(f_alarms+true_neg)

y_pred = model.predict(x_test)
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
print("0.1", verif_pod(y_test, y_pred, 0.1), verif_far(y_test, y_pred, 0.1))
print("0.5", verif_pod(y_test, y_pred, 0.5), verif_far(y_test, y_pred, 0.5))
print("1.0", verif_pod(y_test, y_pred, 1.), verif_far(y_test, y_pred, 1.))
print("2.0", verif_pod(y_test, y_pred, 2.), verif_far(y_test, y_pred, 2.))
print("3.0", verif_pod(y_test, y_pred, 3.), verif_far(y_test, y_pred, 3.))
print("4.0", verif_pod(y_test, y_pred, 4.), verif_far(y_test, y_pred, 4.))
print("5.0", verif_pod(y_test, y_pred, 5.), verif_far(y_test, y_pred, 5.))

tpr = []
tpr.append(1)
fpr = []
fpr.append(1)
for v in [.1,.5,1.,2.,3.,4.,5.]:
    tpr.append(verif_pod(y_test, y_pred, v))
    fpr.append(verif_far(y_test, y_pred, v))

#fpr[6] = 1

print(tpr)
print(fpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         #lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
         lw=2, label='ROC curve (area = %0.2f)' % .3)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
