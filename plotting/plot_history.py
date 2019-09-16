import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pickle
import math


def plot_history(hist_file, var_name, prefix): 
    h = pickle.load(open(hist_file, "rb"))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mse')
    ax1.set_title('Mean Squared Error')
    ax1.set_ylim([0, math.ceil(max(h[var_name]))])
    ax1.plot(h[var_name])
    ax1.plot(h['val_' + var_name])

    fig.legend(['train_' + var_name, 'val_' + var_name], loc='upper right')

    plt.savefig('{}_{}.png'.format(prefix, var_name))


plot_history("../train_history_unet_d_comb_msa_00_10lvels.pkl", "mean_squared_error", "00")
plot_history("../train_history_unet_d_comb_mae_90_10lvels.pkl", "mean_squared_error", "90")
plot_history("../train_history_unet_d_comb_mae_09_10lvels.pkl", "mean_squared_error", "09")

plot_history("../train_history_unet_d_comb_msa_00_10lvels.pkl", "mean_absolute_error", "00")
plot_history("../train_history_unet_d_comb_mae_90_10lvels.pkl", "mean_absolute_error", "90")
plot_history("../train_history_unet_d_comb_mae_09_10lvels.pkl", "mean_absolute_error", "09")

plot_history("../train_history_unet_d_comb_msa_00_10lvels.pkl", "pom", "00")
plot_history("../train_history_unet_d_comb_msa_00_10lvels.pkl", "pom_1", "00")
plot_history("../train_history_unet_d_comb_msa_00_10lvels.pkl", "pom_2", "00")

exit()

plot_history("../train_history_unet_d_comb_mse_00_10lvels.pkl", "pofd", "00")
plot_history("../train_history_unet_d_comb_mse_00_10lvels.pkl", "pofd_1", "00")
plot_history("../train_history_unet_d_comb_mse_00_10lvels.pkl", "pofd_2", "00")

plot_history("../train_history_unet_d_comb_mse_90_10lvels.pkl", "pofd", "90")
plot_history("../train_history_unet_d_comb_mse_90_10lvels.pkl", "pofd_1", "90")
plot_history("../train_history_unet_d_comb_mse_90_10lvels.pkl", "pofd_2", "90")
plot_history("../train_history_unet_d_comb_mse_90_10lvels.pkl", "pom", "90")
plot_history("../train_history_unet_d_comb_mse_90_10lvels.pkl", "pom_1", "90")
plot_history("../train_history_unet_d_comb_mse_90_10lvels.pkl", "pom_2", "90")

plot_history("../train_history_unet_d_comb_mse_09_10lvels.pkl", "pofd", "09")
plot_history("../train_history_unet_d_comb_mse_09_10lvels.pkl", "pofd_1", "09")
plot_history("../train_history_unet_d_comb_mse_09_10lvels.pkl", "pofd_2", "09")
plot_history("../train_history_unet_d_comb_mse_09_10lvels.pkl", "pom", "09")
plot_history("../train_history_unet_d_comb_mse_09_10lvels.pkl", "pom_1", "09")
plot_history("../train_history_unet_d_comb_mse_09_10lvels.pkl", "pom_2", "09")

"""
def plot_mae_history(hist_file, plot_name): 
    h = pickle.load(open(hist_file, "rb"))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mae')
    ax1.set_title('Mean Absolute Error')
    ax1.set_ylim([0, math.ceil(max(h['mean_absolute_error']))])
    ax1.plot(h['mean_absolute_error'])
    ax1.plot(h['val_mean_absolute_error'])

    fig.legend(['train_mae', 'val_mae'], loc='upper right')

    plt.savefig('{}.png'.format(plot_name))


def plot_pod_history(hist_file, plot_name): 
    h = pickle.load(open(hist_file, "rb"))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('probability')
    ax1.set_title('Probability of Detections')
    ax1.set_ylim([0, 1])
    ax1.plot(h['pod'])
    ax1.plot(h['val_pod'])

    fig.legend(['train_pod', 'val_pod'], loc='upper right')

    plt.savefig('{}.png'.format(plot_name))


def plot_far_history(hist_file, plot_name): 
    h = pickle.load(open(hist_file, "rb"))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('probability')
    ax1.set_title('False Alarm Ratio')
    ax1.set_ylim([0, 1])
    ax1.plot(h['far'])
    ax1.plot(h['val_far'])

    fig.legend(['train_far', 'val_far'], loc='upper right')

    plt.savefig('{}.png'.format(plot_name))


def plot_mse_history(hist_file, plot_name): 
    h = pickle.load(open(hist_file), "rb"))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mse')
    ax1.set_title('Mean Square Error')
    ax1.set_ylim([0,5])
    ax1.plot(h['mean_squared_error'])
    ax1.plot(h['val_mean_squared_error'])

    fig.legend(['train_mse', 'val_mse'], loc='upper right')

    plt.save('{}.png'.format(plot_name))
ax2 = ax1.twinx()
ax2.set_ylabel('probability')
ax2.set_ylim([0,1])
ax2.plot(h_mse['val_pod'])
ax2.plot(h_mse['val_far'])

#fig.legend(['train_mse', 'test_mse', 'train_msle', 'test_msle', 'test_pod', 'test_far'], loc='upper right')
#fig.legend(['train_msle', 'test_msle', 'test_pod', 'test_far'], loc='upper right')


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

"""
