import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model

def get_pretrained_oracle(modality= 'bucket'):
    vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(416, 416, 3), pooling=None, classes=1000)
    for layer in vgg.layers:
        layer.trainable = False
    Input_layer = vgg.input
    new = Conv2D(512, 3, activation= 'relu', padding='same', kernel_initializer= 'he_normal')(vgg.layers[-4].output)
    new = Conv2D(512, 3, activation= 'relu', padding='same', kernel_initializer= 'he_normal')(new)
    new = MaxPooling2D((2, 2))(new)
    new = Conv2D(512, 3, activation= 'relu', padding='same', kernel_initializer= 'he_normal')(new)
    new = Conv2D(512, 3, activation= 'relu', padding='same', kernel_initializer= 'he_normal')(new)
    new = MaxPooling2D((2, 2))(new)

    new = Flatten()(new)
    new = Dropout(0.3)(new)
    new = Dense(1024, activation='relu')(new)
    new = Dropout(0.3)(new)
    new = Dense(1024, activation='relu')(new)
    new = Dropout(0.3)(new)
    
    if modality == 'regression':
        new = Dense(2, activation='sigmoid')(new)
    elif modality == 'classification':
        new = Dense(102, activation='softmax')(new)
    else:
        new = Dense(6, activation='softmax')(new)

    return Model(vgg.input, new)