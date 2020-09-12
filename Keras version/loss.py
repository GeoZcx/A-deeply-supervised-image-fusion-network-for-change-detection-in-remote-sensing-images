from keras.losses import binary_crossentropy
import keras.backend as K

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def bce_dice_loss(y_true, y_pred):
    sig_y_true = K.sigmoid(y_true)
    sig_y_pred = K.sigmoid(y_pred)

    dice_loss = 1 - dice_coeff(y_true, y_pred)

    return binary_crossentropy(sig_y_true, sig_y_pred) + dice_loss
