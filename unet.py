from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
import pickle

from categorical_losses import categorical_losses as closs

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

    #outputs = layers.Conv2D(1, (1, 1), activation='relu')(bn18)
    outputs = layers.Conv2D(1, (1, 1))(bn18)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
x = np.load("/data/ERA-Int/10zlevels_min.npy")
print(x.shape)
y = np.load("/data/ERA-Int/tp_min.npy")
print(y.shape, y.min(), y.max(), y.mean(), np.percentile(y, 95), np.percentile(y, 97.5), np.percentile(y, 99))

x_train = x[:16000, :]
x_test = x[16000:, :]
x = None

#Shuffling train dataset for an even training
np.random.seed(0)
idxs_train = np.arange(x_train.shape[0])
np.random.shuffle(idxs_train)
x_train = x_train[idxs_train, :]
y_train = y_train[idxs_train, :]

y = y[:, :, :, None]
y_train = y[:16000, :]
y_test = y[16000:, :]
y = None

print(x_train.shape, y_train.shape)

losses = {'comb3_mse_b5_50': closs.get_diff_comb_mse_loss(1., .5, .5, 0.),
          'comb3_mse_b5_20': closs.get_diff_comb_mse_loss(1., .5, 2., 0.),
          'comb3_mse_b5_40': closs.get_diff_comb_mse_loss(1., .5, 4., 0.),
          'comb3_mse_b5_80': closs.get_diff_comb_mse_loss(1., .5, 8., 0.),
          'comb3_mse_b5_05': closs.get_diff_comb_mse_loss(1., .5, 0., .5),
          'comb3_mse_b5_01': closs.get_diff_comb_mse_loss(1., .5, 0., 1.), 
          'comb3_mse_b5_02': closs.get_diff_comb_mse_loss(1., .5, 0., 2.),
          'comb3_mse_b5_04': closs.get_diff_comb_mse_loss(1., .5, 0., 4.),
          'comb3_mse_b5_08': closs.get_diff_comb_mse_loss(1., .5, 0., 8.),
          'comb3_mse_b5_55': closs.get_diff_comb_mse_loss(1., .5, .5, .5), 
          'comb3_mse_b5_11': closs.get_diff_comb_mse_loss(1., .5, 1., 1.), 
          'comb3_mse_b5_22': closs.get_diff_comb_mse_loss(1., .5, 2., 2.),
          'comb3_mse_b5_44': closs.get_diff_comb_mse_loss(1., .5, 4., 4.),
          'comb3_mse_b5_88': closs.get_diff_comb_mse_loss(1., .5, 8., 8.),
          'comb3_mse_b5_00': closs.get_diff_comb_mse_loss(1., .5, 0., 0.)}


for name, loss in losses.items():
    print(name)
    model = get_unet(loss)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=['mse','mae', closs.get_pom_loss(1.), closs.get_pofd_loss(1.)])
    print(model.summary())
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
    with open('train_history_unet_{}_10lvelsb2.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)
    model.save('unet_{}_10levelsb2.h5'.format(name))
