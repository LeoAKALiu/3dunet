from keras.layers import Input, BatchNormalization, MaxPool3D, Conv3D, UpSampling3D, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from dice import *

def unet(lr):

    inputs = Input((64, 64, 64, 1))

    '''downsample'''
    conv1 = Conv3D(8, 3, 3, padding='same')(inputs)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    conv2 = Conv3D(16, 3, 3, padding='same')(acti1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    maxp1 = MaxPool3D(2)(acti2)


    conv3 = Conv3D(16, 3, 3, padding='same')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    acti3 = Activation('relu')(batc3)
    conv4 = Conv3D(32, 3, 3, padding='same')(acti3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    maxp2 = MaxPool3D(2)(acti4)


    conv5 = Conv3D(32, 3, 3, padding='same')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    acti5 = Activation('relu')(batc5)
    conv6 = Conv3D(64, 3, 3, padding='same')(acti5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    maxp3 = MaxPool3D(2)(acti6)

    conv7 = Conv3D(64, 3, 3, padding='same')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    acti7 = Activation('relu')(batc7)
    conv8 = Conv3D(128, 3, 3, padding='same')(acti7)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)


    '''upsample'''
    upcv1 = Conv3D(64, 3, 3, padding='same', activation='relu')(UpSampling3D(2)(acti8))
    merg1 = Concatenate(axis=-1)([conv6, upcv1])
    conv9 = Conv3D(64, 3, 3, padding='same')(merg1)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    conv10 = Conv3D(64, 3, 3, padding='same')(acti9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    acti10 = Activation('relu')(batc10)

    upcv2 = Conv3D(32, 3, 3, padding='same', activation='relu')(UpSampling3D(2)(acti10))
    merg2 = Concatenate(axis=-1)([conv4, upcv2])
    conv11 = Conv3D(32, 3, padding='same')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    conv12 = Conv3D(32, 3, padding='same')(acti11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    acti12 = Activation('relu')(batc12)

    upcv3 = Conv3D(16, 3, 3, padding='same', activation='relu')(UpSampling3D(2)(acti12))
    merg3 = Concatenate(axis=-1)([conv2, upcv3])
    conv13 = Conv3D(16, 3, padding='same')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    conv14 = Conv3D(16, 3, padding='same')(acti13)
    convol = Conv3D(1, 1, activation='sigmoid')(conv14)


    model = Model(inputs=inputs, outputs=convol)
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model