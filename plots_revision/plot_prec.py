import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sys
sys.path.append('../categorical_losses')

import categorical_losses as closs

red = np.array([255, 252, 250, 247, 244, 242, 239, 236, 234, 231, 229, 226, 223, 221, 218, 215, 213, 210,
                     207, 205, 202, 199, 197, 194, 191, 189, 186, 183, 181, 178, 176, 173, 170, 168, 165, 162,
                     157, 155, 152, 150, 148, 146, 143, 141, 139, 136, 134, 132, 129, 127, 125, 123, 120, 118,
                     116, 113, 111, 109, 106, 104, 102, 100, 97,  95,  93,  90,  88,  86,  83,  81,  79,  77,
                     72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,
                     72,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,
                     73,  78,  83,  87,  92,  97,  102, 106, 111, 116, 121, 126, 130, 135, 140, 145, 150, 154,
                     159, 164, 169, 173, 178, 183, 188, 193, 197, 202, 207, 212, 217, 221, 226, 231, 236, 240,
                     245, 250, 250, 250, 250, 249, 249, 249, 249, 249, 249, 249, 249, 248, 248, 248, 248, 248,
                     248, 248, 247, 247, 247, 247, 247, 247, 247, 246, 246, 246, 246, 246, 246, 246, 246, 245,
                     245, 245, 244, 243, 242, 241, 240, 239, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230,
                     229, 228, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 217, 216, 215, 214,
                     213, 211, 209, 207, 206, 204, 202, 200, 199, 197, 195, 193, 192, 190, 188, 186, 185, 183,
                     181, 179, 178, 176, 174, 172, 171, 169, 167, 165, 164, 162, 160, 158, 157, 155, 153, 151, 150, 146], dtype = np.float) / 255


green = np.array([255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238,
                     237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
                     218, 216, 214, 212, 210, 208, 206, 204, 202, 200, 197, 195, 193, 191, 189, 187, 185, 183,
                     181, 179, 177, 175, 173, 171, 169, 167, 165, 163, 160, 158, 156, 154, 152, 150, 148, 146,
                     142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160,
                     161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179,
                     181, 182, 184, 185, 187, 188, 189, 191, 192, 193, 195, 196, 198, 199, 200, 202, 203, 204,
                     206, 207, 209, 210, 211, 213, 214, 215, 217, 218, 220, 221, 222, 224, 225, 226, 228, 229,
                     231, 232, 229, 225, 222, 218, 215, 212, 208, 205, 201, 198, 195, 191, 188, 184, 181, 178,
                     174, 171, 167, 164, 160, 157, 154, 150, 147, 143, 140, 137, 133, 130, 126, 123, 120, 116,
                     113, 106, 104, 102, 100,  98,  96, 94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74,
                     72,  70,  67,  65,  63,  61,  59,  57,  55,  53,  51,  49,  47,  45,  43,  41,  39,  37,
                     35,  31,  31,  30,  30,  30,  30,  29,  29,  29,  29,  28,  28,  28,  27,  27,  27,  27,
                     26,  26,  26,  26,  25,  25,  25,  25,  24,  24,  24,  23,  23,  23,  23,  22,  22,  22, 22,  21], dtype = np.float) / 255


blue = np.array([255, 255, 255, 254, 254, 254, 254, 253, 253, 253, 253, 253, 252, 252, 252, 252, 252, 251,
                     251, 251, 251, 250, 250, 250, 250, 250, 249, 249, 249, 249, 249, 248, 248, 248, 248, 247,
                     247, 246, 245, 243, 242, 241, 240, 238, 237, 236, 235, 234, 232, 231, 230, 229, 228, 226,
                     225, 224, 223, 221, 220, 219, 218, 217, 215, 214, 213, 212, 211, 209, 208, 207, 206, 204,
                     202, 198, 195, 191, 188, 184, 181, 177, 173, 170, 166, 163, 159, 156, 152, 148, 145, 141,
                     138, 134, 131, 127, 124, 120, 116, 113, 109, 106, 102, 99,  95,  91,  88,  84,  81,  77,
                     70,  71,  71,  72,  72,  73,  74,  74,  75,  75,  76,  77,  77,  78,  78,  79,  80,  80,
                     81,  81,  82,  82,  83,  84,  84,  85,  85,  86,  87,  87,  88,  88,  89,  90,  90,  91,
                     91,  92,  91,  89,  88,  86,  85,  84,  82,  81,  80,  78,  77,  75,  74,  73,  71,  70,
                     69,  67,  66,  64,  63,  62,  60,  59,  58,  56,  55,  53,  52,  51,  49,  48,  47,  45,
                     44,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,
                     41,  41,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,
                     40,  40,  40,  39,  39,  38,  38,  38,  37,  37,  36,  36,  36,  35,  35,  34,  34,  34,
                     33,  33,  32,  32,  31,  31,  31,  30,  30,  29,  29,  29,  28,  28,  27,  27,  27,  26, 26,  25], dtype = np.float) / 255

N = 254
vals = np.ones((N, 4))
vals[:, 0] = red
vals[:, 1] = green
vals[:, 2] = blue
rain = ListedColormap(vals)


x = np.load("/data/ERA-Int/10zlevels_min.npy")
#y = np.log(1+np.load("/data/ERA-Int/tp_min.npy"))
y = np.load("/data/ERA-Int/tp_min.npy")

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

print(idxs[14000:14002])

x = x[idxs, :, :, :]
x_test = x[14000:, :]
x = None

y = y[idxs, :, :, None]
y_test = y[14000:, :]
y = None

print(x_test.shape, y_test.shape)

i = 1
plt.imsave('test_prec_{}.png'.format(i), y_test[i,:,:,0], vmax=20, cmap=rain)

def plot_prec(modelh5, coef, lmbda, mu):
    i = 1
    model = load_model(modelh5, custom_objects={'comb_mse': closs.get_diff_comb_mse_loss(1., lmbda, mu), 'comb_mae': closs.get_diff_comb_mae_loss(1., lmbda, mu), 'pod': closs.get_pod_loss(1.), 'pom': closs.get_pom_loss(1.), 'far': closs.get_far_loss(1.), 'pofd': closs.get_pofd_loss(1.)})
    
    out = model.predict(x_test[i:i+1,:])
    plt.imsave('test_mse{}_{}{}_pred_{}.png'.format(coef, lmbda, mu, i), out[0,:,:,0], vmin=0, vmax=20, cmap=rain)

"""
plot_prec('../unet_comb3_mae_00_10levels.h5', 0, 0)
plot_prec('../unet_comb3_mae_20_10levels.h5', 2, 0)
plot_prec('../unet_comb3_mae_40_10levels.h5', 4, 0)
plot_prec('../unet_comb3_mae_80_10levels.h5', 8, 0)
plot_prec('../unet_comb3_mae_02_10levels.h5', 0, 2)
plot_prec('../unet_comb3_mae_04_10levels.h5', 0, 4)
plot_prec('../unet_comb3_mae_08_10levels.h5', 0, 8)
plot_prec('../unet_comb3_mae_22_10levels.h5', 2, 2)
plot_prec('../unet_comb3_mae_44_10levels.h5', 4, 4)
plot_prec('../unet_comb3_mae_88_10levels.h5', 8, 8)


plot_prec('../unet_comb3_mae_20_10levels.h5', 2, 0)
plot_prec('../unet_comb3_mae_02_10levels.h5', 0, 2)
plot_prec('../unet_comb3_mae_10_10levels.h5', 1, 0)
plot_prec('../unet_comb3_mae_01_10levels.h5', 0, 1)
"""


"""
plot_prec('../unet_comb3_mse_00_10levels.h5', 0, .0, .0)
plot_prec('../unet_comb3_mse_50_10levels.h5', 0, .5, .0)
plot_prec('../unet_comb3_mse_10_10levels.h5', 0, 1., .0)
plot_prec('../unet_comb3_mse_20_10levels.h5', 0, 2., .0)
"""
plot_prec('../unet_comb3_mse_44_10levels.h5', 0, 4., 4.)
plot_prec('../unet_comb3_mse_88_10levels.h5', 0, 8., 8.)
"""
plot_prec('../unet_comb3_mse_05_10levels.h5', 0, .0, .5)
plot_prec('../unet_comb3_mse_01_10levels.h5', 0, .0, 1.)
plot_prec('../unet_comb3_mse_02_10levels.h5', 0, .0, 2.)
plot_prec('../unet_comb3_mse_55_10levels.h5', 0, .5, .5)
plot_prec('../unet_comb3_mse_11_10levels.h5', 0, 1., 1.)
plot_prec('../unet_comb3_mse_22_10levels.h5', 0, 2., 2.)

plot_prec('../unet_comb3_mse_b2_00_10levelsb2.h5', 2, .0, .0)
plot_prec('../unet_comb3_mse_b2_50_10levelsb2.h5', 2, .5, .0)
plot_prec('../unet_comb3_mse_b2_10_10levelsb2.h5', 2, 1., .0)
plot_prec('../unet_comb3_mse_b2_20_10levelsb2.h5', 2, 2., .0)
plot_prec('../unet_comb3_mse_b2_05_10levelsb2.h5', 2, .0, .5)
plot_prec('../unet_comb3_mse_b2_01_10levelsb2.h5', 2, .0, 1.)
plot_prec('../unet_comb3_mse_b2_02_10levelsb2.h5', 2, .0, 2.)
plot_prec('../unet_comb3_mse_b2_55_10levelsb2.h5', 2, .5, .5)
plot_prec('../unet_comb3_mse_b2_11_10levelsb2.h5', 2, 1., 1.)
plot_prec('../unet_comb3_mse_b2_22_10levelsb2.h5', 2, 2., 2.)

plot_prec('../unet_comb3_mse_b5_00_10levelsb2.h5', 5, .0, .0)
plot_prec('../unet_comb3_mse_b5_50_10levelsb2.h5', 5, .5, .0)
plot_prec('../unet_comb3_mse_b5_10_10levelsb2.h5', 5, 1., .0)
plot_prec('../unet_comb3_mse_b5_20_10levelsb2.h5', 5, 2., .0)
plot_prec('../unet_comb3_mse_b5_05_10levelsb2.h5', 5, .0, .5)
plot_prec('../unet_comb3_mse_b5_01_10levelsb2.h5', 5, .0, 1.)
plot_prec('../unet_comb3_mse_b5_02_10levelsb2.h5', 5, .0, 2.)
plot_prec('../unet_comb3_mse_b5_55_10levelsb2.h5', 5, .5, .5)
plot_prec('../unet_comb3_mse_b5_11_10levelsb2.h5', 5, 1., 1.)
plot_prec('../unet_comb3_mse_b5_22_10levelsb2.h5', 5, 2., 2.)
"""
