import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

#from tensorflow.keras import layers
#from tensorflow.keras import models
#from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
#from tensorflow.keras.optimizers import SGD
import numpy as np

from categorical_losses import categorical_losses as closs

def Unet():
    concat_axis = 3
    inputs = tf.keras.layers.Input(shape=(80, 120, 10))

    feats = 32
    bn0 = tf.keras.layers.BatchNormalization(axis=3)(inputs)
    conv1 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    conv1 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)
    conv2 = tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    conv2 = tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv3 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = tf.keras.layers.BatchNormalization(axis=3)(conv3)
    conv3 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = tf.keras.layers.BatchNormalization(axis=3)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn6)

    conv4 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = tf.keras.layers.BatchNormalization(axis=3)(conv4)
    conv4 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = tf.keras.layers.BatchNormalization(axis=3)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 3))(bn8)

    conv5 = tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = tf.keras.layers.BatchNormalization(axis=3)(conv5)
    conv5 = tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = tf.keras.layers.BatchNormalization(axis=3)(conv5)

    up_conv5 = tf.keras.layers.UpSampling2D(size=(2, 3))(bn10)
    up6 = tf.keras.layers.concatenate([up_conv5, conv4], axis=concat_axis)
    conv6 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn11 = tf.keras.layers.BatchNormalization(axis=3)(conv6)
    conv6 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = tf.keras.layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn12)
    up7 = tf.keras.layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = tf.keras.layers.BatchNormalization(axis=3)(conv7)
    conv7 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = tf.keras.layers.BatchNormalization(axis=3)(conv7)

    up_conv7 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn14)
    up8 = tf.keras.layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = tf.keras.layers.BatchNormalization(axis=3)(conv8)
    conv8 = tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = tf.keras.layers.BatchNormalization(axis=3)(conv8)

    up_conv8 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn16)
    up9 = tf.keras.layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(up9)
    bn17 = tf.keras.layers.BatchNormalization(axis=3)(conv9)
    conv9 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = tf.keras.layers.BatchNormalization(axis=3)(conv9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(bn18)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model


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

losses = {'comb_mse2_20': closs.get_diff_comb_mse_loss2(1., 2., 0.),
          'comb_mse2_02': closs.get_diff_comb_mse_loss2(1., 0., 2.),
          'comb_mse2_22': closs.get_diff_comb_mse_loss2(1., 2., 2.)}


@tf.function
def train_step(model, inputs, outputs, optimizer):

  with tf.GradientTape() as t:
    loss = tf.reduce_mean(tf.square(outputs - model(inputs, training=True)))

  grads = t.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train(train_dataset, test_dataset, loss, loss_name, model):
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  pod = closs.get_pod_loss(1.)
  pom = closs.get_pom_loss(1.)
  far = closs.get_far_loss(1.)
  pofd = closs.get_pofd_loss(1.)

  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  pod_loss = tf.keras.metrics.Mean()
  pom_loss = tf.keras.metrics.Mean()
  far_loss = tf.keras.metrics.Mean()
  pofd_loss = tf.keras.metrics.Mean()

  f = open("train_record_{}.out".format(loss_name),"w+")
  f.write('epoch,train_mse,test_mse,test_pod,test_pom,test_far,test_pofd\n')

  for epoch in range(100):
    for (batch, (inputs, outputs)) in enumerate(train_dataset):
      train_step(model, inputs, outputs, optimizer)
      train_loss(loss(model(inputs), outputs))

    for (inputs, outputs) in test_dataset:
      test_loss(loss(model(inputs), outputs))
      pod_loss(pod(model(inputs), outputs))
      pom_loss(pom(model(inputs), outputs))
      far_loss(far(model(inputs), outputs))
      pofd_loss(pofd(model(inputs), outputs))

    template = 'Epoch {}, Loss: {:.4f}, Test Loss: mse: {:.4f}, pod: {:.4f}, pom: {:.4f}, far: {:.4f}, pofd: {:.4f}'
    print(template.format(epoch+1, train_loss.result(), test_loss.result(), pod_loss.result(), pom_loss.result(), far_loss.result(), pofd_loss.result()))
    #print(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(epoch+1, train_loss.result(), test_loss.result(), pod_loss.result(), pom_loss.result(), far_loss.result(), pofd_loss.result()))
    f.flush()

    train_loss.reset_states()
    test_loss.reset_states()

  f.close()
  model.save("unet_{}_10levels.h5".format(loss_name))

# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
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


for name, loss in losses.items():
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  #train_ds = train_ds.shuffle(128)
  train_ds = train_ds.batch(32)
  #train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_ds = test_ds.batch(32)

  model = Unet()
  print(model.summary())
  train(train_ds, test_ds, loss, name, model)
