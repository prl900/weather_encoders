import matplotlib.pyplot as plt
import numpy
import pickle

h_msle = pickle.load(open("train_history_unet_msle_raw_10lvels.pkl", "rb"))
h_mse = pickle.load(open("train_history_unet_mse_raw_10lvels.pkl", "rb"))
h_mae = pickle.load(open("train_history_unet_mae_raw_10lvels.pkl", "rb"))

print(h_mae.keys())

fig, ax1 = plt.subplots()

ax1.set_xlabel('epoch')
ax1.set_ylabel('mse')
ax1.set_title('Mean Square Error')
ax1.set_ylim([0,5])
ax1.plot(h_msle['mean_squared_error'])
ax1.plot(h_msle['val_mean_squared_error'])
ax1.plot(h_mae['mean_squared_error'])
ax1.plot(h_mae['val_mean_squared_error'])
ax1.plot(h_mse['mean_squared_error'])
ax1.plot(h_mse['val_mean_squared_error'])

"""
ax2 = ax1.twinx()
ax2.set_ylabel('probability')
ax2.set_ylim([0,1])
ax2.plot(h_mse['val_pod'])
ax2.plot(h_mse['val_far'])
"""

fig.legend(['train_msle', 'test_msle', 'train_mae', 'test_mae', 'train_mse', 'test_mse'], loc='upper right')
#fig.legend(['train_mse', 'test_mse', 'train_msle', 'test_msle', 'test_pod', 'test_far'], loc='upper right')
#fig.legend(['train_msle', 'test_msle', 'test_pod', 'test_far'], loc='upper right')

plt.show()

exit()

fig, ax1 = plt.subplots()

ax1.set_xlabel('epoch')
ax1.set_ylabel('mse')
ax1.set_title('Mean Absolute Error')
ax1.plot(h_mae['mean_absolute_error'])
ax1.plot(h_mae['val_mean_absolute_error'])
ax1.plot(h_mse['mean_absolute_error'])
ax1.plot(h_mse['val_mean_absolute_error'])

ax2 = ax1.twinx()
ax2.set_ylabel('probability')
ax2.set_ylim([-1,1])
ax2.plot(h_mae['val_far'])
ax2.plot(h_mse['val_far'])

fig.legend(['train_mae', 'test_mae', 'train_mse', 'test_mse', 'far_mae', 'far_mse'], loc='upper right')

plt.show()
exit()

# summarize history for loss
plt.title('Mean Square Error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train_mae', 'test_mae', 'far_mae', 'train_mse', 'test_mse', 'far_mse'], loc='upper right')
plt.show()


plt.plot(h_mae['mean_absolute_error'])
plt.plot(h_mae['val_mean_absolute_error'])
plt.plot(h_mae['val_far'])
plt.plot(h_mse['mean_absolute_error'])
plt.plot(h_mse['val_mean_absolute_error'])
plt.plot(h_mse['val_far'])
plt.title('Mean Absolute Error')

plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train_mae', 'test_mae', 'far_mae', 'train_mse', 'test_mse', 'far_mse'], loc='upper right')
plt.show()
