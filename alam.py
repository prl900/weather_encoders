import numpy as np
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

# Atmospheric Level Activation Mapping (ALAM)

def get_pod_loss(threshold):

    def pod(y_true, y_pred):
        # Probability of detection = Hits / (Hits + Misses)
        hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
    
        return hits/(hits+misses)

    return pod


def get_far_loss(threshold):

    def far(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
    
        return f_alarms/(f_alarms+hits)

    return far


def get_bias_loss(threshold):

    def bias(y_true, y_pred):
        # Bias score = (Hits + False Alarms) / (Hits + Misses)
        hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
    
        return (hits+f_alarms)/(hits+misses)
    
    return bias


def get_ets_loss(threshold):

    def ets(y_true, y_pred):
        # Bias score = (Hits + False Alarms) / (Hits + Misses)
        hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
        true_neg = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
   
        a_ref = ((hits+f_alarms)*(hits+misses))/(hits+f_alarms+misses+true_neg) 

        return (hits-a_ref)/(hits+f_alarms+misses+a_ref)
    
    return ets

def loss(y_true,y_pred):
    return K.mean(K.abs(y_pred-y_true))

def get_rains(codes):
    arr = np.load("data/rain.npy")
    idxs = [airports.index(code) for code in codes]
    return arr[:, idxs].astype(np.float32)

def get_era_full(param, level):
    #arr = np.load("data/{}{}_uint8.npy".format(param, level))[[0], :, :]
    arr = np.load("data/{}{}_uint8.npy".format(param, level))
    #arr = np.load("data/{}{}_low_uint8.npy".format(param, level))[[0], :, :]
    return arr.astype(np.float32) / 256.

def get_alam():
    pod = get_pod_loss(.1)
    far = get_far_loss(.1)
    bias = get_bias_loss(.1)
    ets = get_ets_loss(.1)
    model = load_model('unet.h5', custom_objects={'loss': loss, 'pod': pod, 'far': far, 'bias': bias, 'ets':ets})
    print(len(model.layers))
    print(model.layers[0].get_config())
    print(model.layers[0].get_weights())
    print('-'*24)
    print(model.layers[1].get_config())
    print(model.layers[1].get_weights())
    print('-'*24)
    print(model.layers[2].get_config())
    print(model.layers[2].get_weights())
    weights = np.moveaxis(model.layers[2].get_weights()[0], 2, -1)
    print(weights.shape)
    print(np.sum(np.abs(weights.reshape((-1,10))), axis=0))
    
    print(model.layers[2].get_weights()[0].shape)
    print(model.layers[2].get_weights()[1].shape)
    print('-'*24)
    exit()
    print(model.layers[-5].get_weights()[0].shape)
    print(model.layers[-5].get_weights()[1].shape)
    print(model.layers[-5].get_config())
    print('-'*24)
    print(model.layers[-4].get_weights()[0].shape)
    print(model.layers[-4].get_weights()[1].shape)
    print(model.layers[-4].get_config())
    print('-'*24)
    print(model.layers[-3].get_weights()[0].shape)
    print(model.layers[-3].get_weights()[1].shape)
    print(model.layers[-3].get_config())
    print('-'*24)
    print(model.layers[-2].get_weights()[0].shape)
    print(model.layers[-2].get_weights()[1].shape)
    print(model.layers[-2].get_config())
    print('-'*24)
    print(model.layers[-1].get_weights()[0].shape)
    print(model.layers[-1].get_weights()[1].shape)
    print(model.layers[-1].get_config())
    exit()
    #print(model.layers[-1].get_weights()[0][:, 0].reshape((20,30,256))

    dry_weights = model.layers[-1].get_weights()[0][:, 0].reshape((20,30,256))
    rain_weights = model.layers[-1].get_weights()[0][:, 1].reshape((20,30,256))
    get_output = K.function([model.layers[0].input], [model.layers[-3].output, model.layers[-1].output])

    cam = np.zeros((20,30), dtype=np.float32)
    for depth in range(rain_weights.shape[2]):
        for i in range(30):
            for j in range(20):
                cam[j, i] += rain_weights[j, i, depth]

    cam /= np.max(cam)
    im = Image.fromarray(np.uint8(cm.jet(cam)*255))
    im = im.resize((120,80), Image.ANTIALIAS)
    #heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    im.save("{}_rain.png".format(airport))

if __name__ == '__main__':
    get_alam()
