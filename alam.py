import numpy as np
from keras.models import load_model
import keras.backend as K
from PIL import Image
import matplotlib.cm as cm

# Atmospheric Level Activation Mapping (ALAM)


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
    model = load_model('unet.h5')
    print(len(model.layers))
    print(len(model.layers[-1].get_weights()))
    print('-'*24)
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
