from keras.models import Model

from keras.layers import Input, concatenate, add, Activation, PReLU
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def get_model_unet(output_layers=1, lr=1e-4, input_size =(416, 416, 1), feature_maps = 4, output_type='sigmoid'):
    from keras.layers import Activation, add, BatchNormalization
    
    inputs = Input(input_size)
    
    bn = BatchNormalization()(inputs)
        
    e_conv1 = Conv2D(feature_maps, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(bn)
    e_conv2 = Conv2D(feature_maps, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(e_conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(e_conv2)
    
    e_conv3 = Conv2D(feature_maps * 2, 3, activation = 'relu', padding = 'same' , kernel_initializer='he_normal')(pool1)
    e_conv4 = Conv2D(feature_maps * 2, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(e_conv3) 
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(e_conv4)
    
    e_conv5 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool2)
    e_conv6 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(e_conv5)
    e_conv7 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(e_conv6)

    pool3 = Dropout(0.5)(e_conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(pool3)
    
    e_conv8 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool3)
    e_conv9 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(e_conv8)
    e_conv10 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(e_conv9)
    
    pool4 = Dropout(0.5)(e_conv10)
    pool4 = MaxPooling2D(pool_size=(2, 2))(pool4)
    
    conv11 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool4)
    conv12 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv11)
    conv13 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv12)
    
    up1 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv13))
    up1 = Dropout(0.2)(up1)

    merge1 = concatenate([up1, e_conv10], axis = 3)
    d_conv10 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge1)
    d_conv9 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(d_conv10)
    d_conv8 = Conv2D(feature_maps * 8, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(d_conv9)
    
    up2 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(d_conv8))
    merge2 = concatenate([up2, e_conv7], axis = 3)
    merge2 = Dropout(0.2)(merge2)

    d_conv7 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge2)
    d_conv6 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(d_conv7)
    d_conv5 = Conv2D(feature_maps * 4, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(d_conv6)

    up3 = Conv2D(feature_maps * 2, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(d_conv5))
    merge3 = concatenate([up3,e_conv4], axis = 3)
    merge3 = Dropout(0.2)(merge3)

    d_conv4 = Conv2D(feature_maps * 2, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge3)
    d_conv3 = Conv2D(feature_maps * 2, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(d_conv4)

    up4 = Conv2D(feature_maps, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(d_conv3))
    
    merge4 = concatenate([up4,e_conv2], axis = 3)
    d_conv2 = Conv2D(feature_maps, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge4)
    d_conv1 = Conv2D(feature_maps, 3, activation = 'relu', padding = 'same', kernel_initializer='he_normal')(d_conv2)

    sigmoid = Conv2D(output_layers, (1,1), activation = output_type, kernel_initializer='he_normal')(d_conv1)

    model = Model(inputs=[inputs], outputs=[sigmoid])
    
    return model