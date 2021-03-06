from tensorflow.keras import backend as K
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


def get_pom_loss(threshold):
    def pom(y_true, y_pred):
        # Probability of missing = Misses / (Hits + Misses)
        hits = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        misses = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))

        return misses / (hits + misses)

    return pom


def get_far_loss(threshold):
    def far(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(
            K.cast(tf.math.logical_and(K.greater(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        f_alarms = K.sum(
            K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))

        return f_alarms / (f_alarms + hits)

    return far


def get_pofd_loss(threshold):
    def pofd(y_true, y_pred):

        # Probability of False Detection = False Alarms / (False Alarms + True Negatives)
        f_alarms = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.greater(y_pred, threshold)), dtype='float32'))
        true_neg = K.sum(K.cast(tf.math.logical_and(K.less(y_true, threshold), K.less(y_pred, threshold)), dtype='float32'))

        return f_alarms / (f_alarms + true_neg)

    return pofd


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
        # Probability of detection = Misses / (Hits + Misses)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        pod = hits / (hits + misses)
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        return pod + mae

    return pod_mae

def get_diff_pod_mse_loss(threshold):
    def pod_mse(y_true, y_pred):
        # Probability of detection = Misses / (Hits + Misses)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        pod = hits / (hits + misses)
        mse = K.mean(K.square(y_pred - y_true), axis=-1)

        return 0.1*pod + mse

    return pod_mse


def get_diff_pom_mae_loss(threshold):
    def pom_mae(y_true, y_pred):
        # Probability of missing = Misses / (Hits + Misses)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        pom = misses / (hits + misses)
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        return pom + mae

    return pom_mae

def get_diff_pom_mse_loss(threshold, pom_coef):
    def pom_mse(y_true, y_pred):
        # Probability of missing = Misses / (Hits + Misses)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        misses = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        pom = misses / (hits + misses)
        mse = K.mean(K.square(y_pred - y_true), axis=-1)

        return pom_coef*pom + mse

    return pom_mse

def get_diff_pofd_mae_loss(threshold, pofd_coef):
    def pofd_mae(y_true, y_pred):
        # Probability of false detection = False Alarms / (False Alarms + Hits)
        true_neg = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        pofd = f_alarms / (f_alarms + true_neg) 
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        return pofd_coef*pofd + mae

    return pofd_mae


def get_diff_pofd_mse_loss(threshold, pofd_coef):
    def pofd_mse(y_true, y_pred):
        # Probability of false detection = False Alarms / (False Alarms + Hits)
        true_neg = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid((-1 * y_pred) - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        pofd = f_alarms / (f_alarms + true_neg) 
        mse = K.mean(K.square(y_pred - y_true), axis=-1)

        return pofd_coef*pofd + mse

    return pofd_mse

def get_diff_far_mae_loss(threshold, far_coef):
    def far_mae(y_true, y_pred):
        # False Alarm Ratio score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        far = f_alarms / (f_alarms + hits)
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        return far_coef*far + mae

    return far_mae


def get_diff_far_mse_loss(threshold, far_coef):
    def far_mse(y_true, y_pred):
        # False Alarm Rate score = False Alarms / (False Alarms + Hits)
        hits = K.sum(K.cast(K.sigmoid(y_true - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        f_alarms = K.sum(K.cast(K.sigmoid((-1 * y_true) - threshold) * K.sigmoid(y_pred - threshold), dtype='float32'))
        far = f_alarms / (f_alarms + hits)
        mse = K.mean(K.square(y_pred - y_true), axis=-1)
        
        return far_coef*far + mse

    return far_mse


def get_diff_comb_mae_loss(threshold, beta, pom_coef, pofd_coef):
    def comb_mae(y_true, y_pred):
        hits = K.sum(K.cast(K.greater(y_true, threshold), dtype='float32') * K.sigmoid(beta * (y_pred - threshold)))
        misses = K.sum(K.cast(K.greater(y_true, threshold), dtype='float32') * K.sigmoid(beta * (threshold - y_pred)))
        pom = misses / (hits + misses)
        f_alarms = K.sum(K.cast(K.less(y_true, threshold), dtype='float32') * K.sigmoid(beta * (y_pred - threshold)))
        true_neg = K.sum(K.cast(K.less(y_true, threshold), dtype='float32') * K.sigmoid(beta * (threshold - y_pred)))
        pofd = f_alarms / (f_alarms + true_neg) 
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        return pom_coef*pom + pofd_coef*pofd + mae

    return comb_mae

def get_diff_comb_mse_loss(threshold, beta, pom_coef, pofd_coef):
    def comb_mse(y_true, y_pred):
        hits = K.sum(K.cast(K.greater(y_true, threshold), dtype='float32') * K.sigmoid(beta * (y_pred - threshold)))
        misses = K.sum(K.cast(K.greater(y_true, threshold), dtype='float32') * K.sigmoid(beta * (threshold - y_pred)))
        pom = misses / (hits + misses)
        f_alarms = K.sum(K.cast(K.less(y_true, threshold), dtype='float32') * K.sigmoid(beta * (y_pred - threshold)))
        true_neg = K.sum(K.cast(K.less(y_true, threshold), dtype='float32') * K.sigmoid(beta * (threshold - y_pred)))
        pofd = f_alarms / (f_alarms + true_neg) 
        mse = K.mean(K.square(y_pred - y_true), axis=-1)

        return pom_coef*pom + pofd_coef*pofd + mse

    return comb_mse

