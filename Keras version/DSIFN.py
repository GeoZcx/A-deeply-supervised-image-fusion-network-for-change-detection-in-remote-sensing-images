
# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

from keras import applications
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D,Conv2DTranspose,Subtract,multiply,add,UpSampling2D,PReLU
from keras import  layers
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from attention_module import channel_attention, spatial_attention,get_spatial_attention_map

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same',with_activation = False):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides)(x)
    x=PReLU()(x)
    x=BatchNormalization(axis=3)(x)
    x=Dropout(rate = 0.6)(x)
    if with_activation == True:
        x = Activation('relu')(x)
    return x

def vgg16():
    vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(512,512,3))
    model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block5_conv3').output)
    model.trainable=False
    return model

def DSIFN():
    #DFEN accepts inputs in size of 512*512*3
    vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(512,512,3))

    b5c3_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block5_conv3').output)
    b5c3_model.trainable=False

    b4c3_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block4_conv3').output)
    b4c3_model.trainable=False

    b3c3_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block3_conv3').output)
    b3c3_model.trainable=False

    b2c2_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block2_conv2').output)
    b2c2_model.trainable=False

    b1c2_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block1_conv2').output)
    b1c2_model.trainable=False

    input_t1 = layers.Input((512,512,3), name='Input_t1')
    input_t2 = layers.Input((512,512,3), name='Input_t2')

    t1_b5c3 = b5c3_model(input_t1)
    t2_b5c3 = b5c3_model(input_t2)

    t1_b4c3 = b4c3_model(input_t1)
    t2_b4c3 = b4c3_model(input_t2)

    t1_b3c3 = b3c3_model(input_t1)
    t2_b3c3 = b3c3_model(input_t2)

    t1_b2c2 = b2c2_model(input_t1)
    t2_b2c2 = b2c2_model(input_t2)

    t1_b1c2 = b1c2_model(input_t1)
    t2_b1c2 = b1c2_model(input_t2)

    concat_b5c3 = concatenate([t1_b5c3, t2_b5c3], axis=3) #channel 1024
    x = Conv2d_BN(concat_b5c3,512, 3)
    x = Conv2d_BN(x,512,3)
    attention_map_1 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_1])
    x = BatchNormalization(axis=3)(x)

    #branche1
    branch_1 =Conv2D(1, kernel_size=1, activation='sigmoid', padding='same',name='output_32')(x)

    x = Conv2DTranspose(512, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b4c3,t2_b4c3],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,512,3)
    x = Conv2d_BN(x,256,3)
    x = Conv2d_BN(x,256,3)
    attention_map_2 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_2])
    x = BatchNormalization(axis=3)(x)

    #branche2
    branch_2 =Conv2D(1, kernel_size=1, activation='sigmoid', padding='same',name='output_64')(x)

    x = Conv2DTranspose(256, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b3c3,t2_b3c3],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,256,3)
    x = Conv2d_BN(x,128,3)
    x = Conv2d_BN(x, 128, 3)
    attention_map_3 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_3])
    x = BatchNormalization(axis=3)(x)

    #branche3
    branch_3 =Conv2D(1, kernel_size=1, activation='sigmoid', padding='same',name='output_128')(x)

    x = Conv2DTranspose(128, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b2c2,t2_b2c2],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,128,3)
    x = Conv2d_BN(x,64,3)
    x = Conv2d_BN(x, 64, 3)
    attention_map_4 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_4])
    x = BatchNormalization(axis=3)(x)

    #branche4
    branch_4 =Conv2D(1, kernel_size=1, activation='sigmoid', padding='same',name='output_256')(x)

    x = Conv2DTranspose(64, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(x)
    x = concatenate([x,t1_b1c2,t2_b1c2],axis=3)
    x = channel_attention(x)
    x = Conv2d_BN(x,64,3)
    x = Conv2d_BN(x,32,3)
    x = Conv2d_BN(x, 16, 3)
    attention_map_5 = get_spatial_attention_map(x)
    x = multiply([x, attention_map_5])

    # branche5
    branch_5 =Conv2D(1, kernel_size=1, activation='sigmoid', padding='same',name='output_512')(x)

    DSIFN = Model(inputs=[input_t1,input_t2], outputs=[branch_1,branch_2,branch_3,branch_4,branch_5])

    return DSIFN
