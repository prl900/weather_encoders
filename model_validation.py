import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

def comb_mae(y_true, y_pred):
    threshold = .5
    # False Alarm Rate score = False Alarms / (False Alarms + Hits)
    hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
    misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
    f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))

    return (hits+misses)/hits + f_alarms/(f_alarms+hits) + K.mean(K.abs(y_pred-y_true), axis=-1)

def pod_mae(y_true, y_pred):
    threshold = .5
    # Probability of detection = Hits / (Hits + Misses)
    hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
    misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))

    return (hits+misses)/hits + K.mean(K.abs(y_pred - y_true), axis=-1)

def pod(y_true, y_pred):
    threshold = .5
    # Probability of detection = Hits / (Hits + Misses)
    hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    misses = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))

    return hits/(hits+misses)

def far(y_true, y_pred):
    threshold = .5
    # False Alarm Rate score = False Alarms / (False Alarms + Hits)
    hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    
    return f_alarms/(f_alarms+hits)

def bias(y_true, y_pred):
    threshold = .5
    # Bias score = (Hits + False Alarms) / (Hits + Misses)
    hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    misses = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
    
    return (hits+f_alarms)/(hits+misses)

def ets(y_true, y_pred):
    threshold = .5
    # Bias score = (Hits + False Alarms) / (Hits + Misses)
    hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    misses = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
    true_neg = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
   
    a_ref = ((hits+f_alarms)*(hits+misses))/(hits+f_alarms+misses+true_neg) 

    return (hits-a_ref)/(hits+f_alarms+misses+a_ref)


x = np.load("/data/ERA-Int/10zlevels_min.npy")

y = np.load("/data/ERA-Int/tp_min.npy")
#y = np.log(1+np.load("/data/ERA-Int/tp_min.npy"))

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :, :]
x_test = x[14000:, :]
x = None

y = y[idxs, :, :, None]
y_test = y[14000:, :]
y = None

print("Done 1")

model = load_model('unet_mae_raw_10levels.h5', custom_objects={'pod': pod, 'far': far, 'ets': ets, 'bias': bias})

out = model.predict(x_test)
print(out.shape)
print(y_test.shape)
print("mse:", np.mean(np.square(y_test - out)))
