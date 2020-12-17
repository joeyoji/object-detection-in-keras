import tensorflow as tf


class SMOOTH_L1_LOSS:
    """
    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py

    Paper References:
        - Girshick, R. (2015). Fast-RCNN. https://arxiv.org/pdf/1504.08083.pdf
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """

    def compute(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * tf.square(y_true - y_pred)
        loss = tf.where(abs_loss < 1, square_loss, abs_loss - 0.5)
        return tf.reduce_sum(loss, axis=-1)