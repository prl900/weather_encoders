from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.optimizers import Adam,SGD
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import numpy as np
import pickle

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
    hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, .1), K.greater(y_pred, .1)), dtype='float32'))
    f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, .1), K.greater(y_pred, .1)), dtype='float32'))
    
    #return K.mean(K.square(y_pred-y_true)) + 10*(f_alarms/(f_alarms+hits))
    #return K.mean(K.abs(y_pred-y_true)) + 10*(f_alarms/(f_alarms+hits))
    #return 10*(f_alarms/(f_alarms+hits))
    return K.mean(K.log(1+K.abs(y_pred - y_true)), axis=-1)

def get_unet(loss):
    concat_axis = 3
    inputs = layers.Input(shape = (80, 120, 10))

    feats = 32
    bn0 = BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6)

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 3))(bn8)

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = BatchNormalization(axis=3)(conv5)
    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = BatchNormalization(axis=3)(conv5)

    up_conv5 = layers.UpSampling2D(size=(2, 3))(bn10)
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn11 = BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12)
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = BatchNormalization(axis=3)(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(bn14)
    up8 = layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = BatchNormalization(axis=3)(conv8)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = BatchNormalization(axis=3)(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(bn16)
    up9 = layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(up9)
    bn17 = BatchNormalization(axis=3)(conv9)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = BatchNormalization(axis=3)(conv9)

    conv10 = layers.Conv2D(1, (1, 1))(bn18)
    #bn19 = BatchNormalization(axis=3)(conv10)

    model = models.Model(inputs=inputs, outputs=conv10)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    pod = get_pod_loss(.1)
    far = get_far_loss(.1)
    bias = get_bias_loss(.1)
    ets = get_ets_loss(.1)
    #model.compile(loss=loss, optimizer=sgd, metrics=['mse','mae'])
    model.compile(loss=loss, optimizer=sgd, metrics=['mse','mae', pod, far, bias, ets])
    #model.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=['mse'])
    print(model.summary())

    return model


# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
#x = x[:, :, :, ]
#x = x[:, :, :, [0,2,5]]
#x = np.load("/data/ERA-Int/10zlevels.npy")[:, :, :, [0,2,5]]
x = np.load("/data/ERA-Int/10zlevels_min.npy")
print(x.shape)
y = np.log(1+np.clip(100*np.load("/data/ERA-Int/tp_min.npy"), 0, 18))

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :, :]
x_train = x[:14000, :]
x_test = x[14000:, :]

y = y[idxs, :, :, None]
y_train = y[:14000, :]
y_test = y[14000:, :]

print(x_train.shape, y_train.shape)

model = get_unet('mae')
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
with open('train_history_unet_far_10lvels.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save('unet_far_10levels.h5')

"""
model = get_unet('mse')
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
with open('train_history_unet_mse_10lvels.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save('unet_mse_10levels.h5')
"""
