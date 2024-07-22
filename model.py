from tensorflow.keras import layers, models

def conv_bn(x, filters, bn=True):
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same')(x)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    l = conv_bn(input_layer, 32)
    c1 = conv_bn(l, 32)
    l = layers.MaxPool2D(strides=(2,2))(c1)
    l = conv_bn(l, 64)
    c2 = conv_bn(l, 64)
    l = layers.MaxPool2D(strides=(2,2))(c2)
    l = conv_bn(l, 128)
    c3 = conv_bn(l, 128)
    l = layers.MaxPool2D(strides=(2,2))(c3)
    l = conv_bn(l, 128)
    c4 = conv_bn(l, 128)

    l = layers.concatenate([layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(c4), c3], axis=-1)
    l = conv_bn(l, 64)
    l = conv_bn(l, 64)
    l = layers.concatenate([layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(l), c2], axis=-1)
    l = conv_bn(l, 128)
    l = conv_bn(l, 128)
    l = layers.concatenate([layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(l), c1], axis=-1)
    l = conv_bn(l, 128)
    l = conv_bn(l, 128)
    output_layer = layers.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)

    model = models.Model(input_layer, output_layer)
    return model
