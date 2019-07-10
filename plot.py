from plotting import plot_history

plot_history.plot_mse_history("train_history_unet_mae_raw_10lvels.pkl", "mse_mae_raw")
plot_history.plot_mae_history("train_history_unet_mae_raw_10lvels.pkl", "mae_mae_raw")
plot_history.plot_pod_history("train_history_unet_mae_raw_10lvels.pkl", "pod_mae_raw")
plot_history.plot_far_history("train_history_unet_mae_raw_10lvels.pkl", "far_mae_raw")

plot_history.plot_mse_history("train_history_unet_mae_10lvels.pkl", "mse_mae")
plot_history.plot_mae_history("train_history_unet_mae_10lvels.pkl", "mae_mae")
plot_history.plot_pod_history("train_history_unet_mae_10lvels.pkl", "pod_mae")
plot_history.plot_far_history("train_history_unet_mae_10lvels.pkl", "far_mae")

plot_history.plot_mse_history("train_history_unet_mse_raw_10lvels.pkl", "mse_mse_raw")
plot_history.plot_mae_history("train_history_unet_mse_raw_10lvels.pkl", "mae_mse_raw")
plot_history.plot_pod_history("train_history_unet_mse_raw_10lvels.pkl", "pod_mse_raw")
plot_history.plot_far_history("train_history_unet_mse_raw_10lvels.pkl", "far_mse_raw")

plot_history.plot_mse_history("train_history_unet_mse_10lvels.pkl", "mse_mse")
plot_history.plot_mae_history("train_history_unet_mse_10lvels.pkl", "mae_mse")
plot_history.plot_pod_history("train_history_unet_mse_10lvels.pkl", "pod_mse")
plot_history.plot_far_history("train_history_unet_mse_10lvels.pkl", "far_mse")
