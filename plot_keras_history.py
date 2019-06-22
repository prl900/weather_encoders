import matplotlib.pyplot as plt
import numpy
import pickle

history = pickle.load(open("train_history_unet_mse_7lvels.pkl", "rb"))

print(history.keys())
# summarize history for loss
plt.plot(history['loss'])#[:20])
plt.plot(history['val_loss'])#[:20])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
