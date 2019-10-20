import keras.backend as K

def dice_coef_sig(y_true, y_pred):
    '''
    Computes dice score for binary outputs. +1 for 1&1, 0 otherwise.
    '''
    # flatten the values
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # if both of them does not have 1 then count them into the intersection
    intersection = K.sum(y_true_f * y_pred_f)
    # smooth is used to avoid nan by images without 1's
    smooth = 1e-4
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)