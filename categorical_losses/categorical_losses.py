from keras import backend as K
import tensorflow as tf

# Non differentiable losses

def get_pod_loss(threshold):
    def pod(y_true, y_pred):
        # Probability of detection = Hits / (Hits + Misses)
        hits = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))

        return hits / (hits + misses)

    return pod

def get_far_loss(threshold):
    def far(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(
            K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))

        return f_alarms / (f_alarms + hits)

    return far

def get_bias_loss(threshold):
    def bias(y_true, y_pred):
        # Bias score = (Hits + False Alarms) / (Hits + Misses)
        hits = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(
            K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))

        return (hits + f_alarms) / (hits + misses)

    return bias

def get_ets_loss(threshold):
    def ets(y_true, y_pred):
        # Bias score = (Hits + False Alarms) / (Hits + Misses)
        hits = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(
            K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))
        true_neg = K.sum(
            K.cast(tf.math.logical_and(K.less(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))

        a_ref = ((hits + f_alarms) * (hits + misses)) / (hits + f_alarms + misses + true_neg)

        return (hits - a_ref) / (hits + f_alarms + misses + a_ref)

    return ets


# Differentiable losses

def get_diff_pod_mae_loss(threshold):
    def pod_mae(y_true, y_pred):
        # Probability of detection = Hits / (Hits + Misses)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))

        return (hits + misses) / hits + K.mean(K.abs(y_pred - y_true), axis=-1)

    return pod_mae


def get_diff_pod_mse_loss(threshold):
    def pod_mse(y_true, y_pred):
        # Probability of detection = Hits / (Hits + Misses)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))

        return (hits + misses) / hits + K.mean(K.square(y_pred - y_true), axis=-1)

    return pod_mse


def get_diff_far_mae_loss(threshold, coef):
    def far_mae(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))

        return (coef * f_alarms / (f_alarms + hits)) + K.mean(K.abs(y_pred - y_true), axis=-1)

    return far_mae


def get_diff_far_mse_loss(threshold):
    def far_mse(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))

        return f_alarms / (f_alarms + hits) + K.mean(K.square(y_pred - y_true), axis=-1)

    return far_mse


def get_diff_comb_mae_loss(threshold):
    def comb_mae(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))

        return (hits + misses) / hits + f_alarms / (f_alarms + hits) + K.mean(K.abs(y_pred - y_true), axis=-1)

    return comb_mae


def get_diff_comb_mse_loss(threshold):
    def comb_mse(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))

        return (hits + misses) / hits + f_alarms / (f_alarms + hits) + K.mean(K.square(y_pred - y_true), axis=-1)

    return comb_mse


def loss(y_true, y_pred):
    hits = K.sum(K.cast(tf.math.logical_and(K.greater(y_true, .1), K.greater(y_pred, .1)), dtype='float32'))
    f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, .1), K.greater(y_pred, .1)), dtype='float32'))

    # return K.mean(K.square(y_pred-y_true)) + 10*(f_alarms/(f_alarms+hits))
    # return K.mean(K.abs(y_pred-y_true)) + 10*(f_alarms/(f_alarms+hits))
    # return 10*(f_alarms/(f_alarms+hits))
    return K.mean(K.log(1 + K.abs(y_pred - y_true)), axis=-1)


def rms_loss(y_true, y_pred):
    return K.mean(K.square(K.log(y_pred + 6.4) - K.log(y_true + 1.4)), axis=-1)


def rmsf_loss(y_true, y_pred):
    # RMSF as in:
    # https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1017/S1350482798000577

    # return K.exp(K.sqrt(K.mean(K.log(K.square(((y_pred+.1) / (y_true+.1)))), axis=-1)))
    return K.mean(K.square(K.log(y_pred + 1.5) - K.log(y_true + 1.4)), axis=-1)