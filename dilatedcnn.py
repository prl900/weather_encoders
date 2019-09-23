from keras import layers
from keras import models
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
import numpy as np
import pickle
from categorical_losses import categorical_losses as closs

def get_unet(loss, feats):
    concat_axis = 3
    inputs = layers.Input(shape = (80, 120, 10))

    #feats = 64
    conv = layers.Conv2D(feats, (5, 5), activation='relu', padding='same')(inputs)
    bn = BatchNormalization(axis=3)(conv)
    conv = layers.Conv2D(feats, (5, 5), activation='relu', padding='same')(bn)
    bn = BatchNormalization(axis=3)(conv)
    
    conv = layers.Conv2D(2*feats, (5, 5), activation='relu', padding='same', dilation_rate=(2, 2))(bn)
    bn = BatchNormalization(axis=3)(conv)
    conv = layers.Conv2D(2*feats, (5, 5), activation='relu', padding='same', dilation_rate=(2, 2))(bn)
    bn = BatchNormalization(axis=3)(conv)

    conv = layers.Conv2D(4*feats, (5, 5), activation='relu', padding='same', dilation_rate=(3, 3))(bn)
    bn = BatchNormalization(axis=3)(conv)
    conv = layers.Conv2D(4*feats, (5, 5), activation='relu', padding='same', dilation_rate=(3, 3))(bn)
    bn = BatchNormalization(axis=3)(conv)

    conv = layers.Conv2D(8*feats, (5, 5), activation='relu', padding='same', dilation_rate=(4, 4))(bn)
    bn = BatchNormalization(axis=3)(conv)
    conv = layers.Conv2D(8*feats, (5, 5), activation='relu', padding='same', dilation_rate=(4, 4))(bn)
    bn = BatchNormalization(axis=3)(conv)
    
    """
    conv = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same', dilation_rate=(16, 16))(bn)
    bn = BatchNormalization(axis=3)(conv)
    conv = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same', dilation_rate=(16, 16))(bn)
    bn = BatchNormalization(axis=3)(conv)
    """
    
    conv = layers.Conv2D(1, (1, 1))(bn)

    model = models.Model(inputs=inputs, outputs=conv)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=sgd, metrics=['mse','mae'])
    print(model.summary())

    return model


# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
#x = x[:, :, :, ]
#x = x[:, :, :, [0,2,5]]
#x = np.load("/data/ERA-Int/10zlevels.npy")[:, :, :, [0,2,5]]
x = np.load("/data/ERA-Int/10zlevels_min.npy")
print(x.shape)
y = np.load("/data/ERA-Int/tp_min.npy")
print(y.shape, y.min(), y.max(), y.mean(), np.percentile(y, 95), np.percentile(y, 97.5), np.percentile(y, 99))

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
"""
losses = {'pom_mae_raw': closs.get_diff_pom_mae_loss(.5), 'pofd_mae_raw': closs.get_diff_pofd_mae_loss(.5),'pom_mse_raw': closs.get_diff_pom_mse_loss(.5), 'pofd_mse_raw': closs.get_diff_pofd_mse_loss(.5), 'mae_raw': 'mae', 'mse_raw': 'mse'}

for name, loss in losses.items():
    print(name)
    model = get_unet(loss)
    history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test))
    with open('train_history_unet_{}_10lvels.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)
    model.save('unet_{}_10levels.h5'.format(name))

y_train = np.log(1+y_train)
y_test = np.log(1+y_test)
"""

#losses = {'pom_mae': closs.get_diff_pom_mae_loss(.5), 'pofd_mae': closs.get_diff_pofd_mae_loss(.5),'pom_mse': closs.get_diff_pom_mse_loss(.5), 'pofd_mse': closs.get_diff_pofd_mse_loss(.5), 'mae': 'mae', 'mse': 'mse'}
#losses = {'far_mae': closs.get_diff_far_mae_loss(.5),'pom_mse': closs.get_diff_pom_mse_loss(.5), 'far_mse': closs.get_diff_far_mse_loss(.5), 'pofd_mse': closs.get_diff_pofd_mse_loss(.5), 'comb_mae': closs.get_diff_comb_mae_loss(.5), 'comb_mse': closs.get_diff_comb_mse_loss(.5, .1, .1)}
#losses = {'comb_mse_00': closs.get_diff_comb_mse_loss(1., .0, .0), 'comb_mse_02': closs.get_diff_comb_mse_loss(1., .0, .2), 'comb_mse_04': closs.get_diff_comb_mse_loss(1., .0, .4), 'comb_mse_08': closs.get_diff_comb_mse_loss(1., .0, .8), 'comb_mse_20': closs.get_diff_comb_mse_loss(1., .2, .0), 'comb_mse_40': closs.get_diff_comb_mse_loss(1., .4, .0), 'comb_mse_80': closs.get_diff_comb_mse_loss(1., .8, .0), 'comb_mse_55': closs.get_diff_comb_mse_loss(1., .5, .5)}
#losses = {'d_comb_mae_d20': closs.get_diff_comb_mae_loss(1., 2., .0), 'd_comb_mae_0d2': closs.get_diff_comb_mae_loss(1., .0, 2.)}
losses = {'comb_mae_00': closs.get_diff_comb_mae_loss(1., 0., 0.), 
          'comb_mae_10': closs.get_diff_comb_mae_loss(1., 1., 0.),
          'comb_mae_20': closs.get_diff_comb_mae_loss(1., 2., 0.),
          'comb_mae_40': closs.get_diff_comb_mae_loss(1., 4., 0.),
          'comb_mae_80': closs.get_diff_comb_mae_loss(1., 8., 0.),
          'comb_mae_01': closs.get_diff_comb_mae_loss(1., 0., 1.), 
          'comb_mae_02': closs.get_diff_comb_mae_loss(1., 0., 2.),
          'comb_mae_04': closs.get_diff_comb_mae_loss(1., 0., 4.),
          'comb_mae_08': closs.get_diff_comb_mae_loss(1., 0., 8.),
          'comb_mae_11': closs.get_diff_comb_mae_loss(1., 1., 1.), 
          'comb_mae_22': closs.get_diff_comb_mae_loss(1., 2., 2.),
          'comb_mae_44': closs.get_diff_comb_mae_loss(1., 4., 4.),
          'comb_mae_88': closs.get_diff_comb_mae_loss(1., 8., 8.),
          'comb_mse_00': closs.get_diff_comb_mse_loss(1., 0., 0.), 
          'comb_mse_10': closs.get_diff_comb_mse_loss(1., 1., 0.),
          'comb_mse_20': closs.get_diff_comb_mse_loss(1., 2., 0.),
          'comb_mse_40': closs.get_diff_comb_mse_loss(1., 4., 0.),
          'comb_mse_80': closs.get_diff_comb_mse_loss(1., 8., 0.),
          'comb_mse_01': closs.get_diff_comb_mse_loss(1., 0., 1.), 
          'comb_mse_02': closs.get_diff_comb_mse_loss(1., 0., 2.),
          'comb_mse_04': closs.get_diff_comb_mse_loss(1., 0., 4.),
          'comb_mse_08': closs.get_diff_comb_mse_loss(1., 0., 8.),
          'comb_mse_11': closs.get_diff_comb_mse_loss(1., 1., 1.), 
          'comb_mse_22': closs.get_diff_comb_mse_loss(1., 2., 2.),
          'comb_mse_44': closs.get_diff_comb_mse_loss(1., 4., 4.),
          'comb_mse_88': closs.get_diff_comb_mse_loss(1., 8., 8.)}

for feats in [16, 32, 64]:
    name = 'mse_{}'.format(feats)
    model = get_unet('mse', feats)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    history = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_data=(x_test, y_test))
    with open('train_history_unet_{}_10lvels.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)
    model.save('unet_{}_10levels.h5'.format(name))

exit()

for name, loss in losses.items():
    print(name)
    model = get_unet(loss)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    history = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_data=(x_test, y_test))
    with open('train_history_unet_{}_10lvels.pkl'.format(name), 'wb') as f:
        pickle.dump(history.history, f)
    model.save('unet_{}_10levels.h5'.format(name))
