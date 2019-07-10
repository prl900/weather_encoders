import numpy as np

def sr(y_obs, y_pred, threshold=.5):

    # Success Ratio = Hits / (Hits + False Alarms)
    hits = np.sum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    f_alarms = np.sum(np.logical_and((y_obs < threshold), (y_pred > threshold)))

    return hits / (hits + f_alarms)

def pod(y_obs, y_pred, threshold=.5):

    # Probability of detection = Hits / (Hits + Misses)
    hits = np.sum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    misses = np.sum(np.logical_and((y_obs > threshold), (y_pred < threshold)))

    return hits / (hits + misses)


def far(y_obs, y_pred, threshold=.5):

    # False Alarm Ratio = False Alarms / (Hits + False Alarms)
    hits = np.sum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    f_alarms = np.sum(np.logical_and((y_obs < threshold), (y_pred > threshold)))

    return f_alarms / (f_alarms + hits)


def pofd(y_obs, y_pred, threshold=.5):

    # Probability of False Detection
    true_neg = np.sum(np.logical_and((y_obs < threshold), (y_pred < threshold)))
    f_alarms = np.sum(np.logical_and((y_obs < threshold), (y_pred > threshold)))

    return f_alarms / (f_alarms + true_neg)


def bias(y_obs, y_pred, threshold=.5):

    # Bias score = (Hits + False Alarms) / (Hits + Misses)
    hits = np.sum(np.logical_and((y_obs > threshold), (y_pred > threshold)))
    f_alarms = np.sum(np.logical_and((y_obs < threshold), (y_pred > threshold)))
    misses = np.sum(np.logical_and((y_obs > threshold), (y_pred < threshold)))

    return (hits + f_alarms) / (hits + misses)
