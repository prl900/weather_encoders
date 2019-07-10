import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from verification import verification as vf
from categorical_losses import categorical_losses as closs

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

model = load_model('unet_mae_raw_10levels.h5', custom_objects={'pod': closs.get_diff_pod_loss(.5), 'far': closs.get_diff_far_loss(.5), 'bias': closs.get_diff_bias_loss(.5)})

out = model.predict(x_test)
print(out.shape)
print(y_test.shape)
print("mse:", np.mean(np.square(y_test - out)))
print("mse:", vf.pod(y_test, out))
