import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """Constructs a basic contracting block for U-Net."""
    conv = Conv2D(
        n_filters, kernel_size=3, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(inputs)
    conv = Conv2D(
        n_filters, kernel_size=3, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(conv)
    
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """Constructs an expanding up-sample block for U-Net."""
    up = Conv2DTranspose(
        n_filters, kernel_size=3, activation='relu', strides=(2, 2), padding='same'
    )(expansive_input)
  
    merge = concatenate([up, contractive_input], axis=3)
    
    conv = Conv2D(
        n_filters, kernel_size=3, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(merge)
    conv = Conv2D(
        n_filters, kernel_size=3, activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(conv)

    return conv

def build_unet_model(input_shape=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Builds the full U-Net architecture.
    
    Args:
        input_shape: Tuple (height, width, channels)
        n_filters: Base number of filters for the first convolution block.
        n_classes: Number of distinct segmentation classes.
    """
    inputs = Input(input_shape)
    
    # Contracting path
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], 2 * n_filters)
    cblock3 = conv_block(cblock2[0], 4 * n_filters)
    cblock4 = conv_block(cblock3[0], 8 * n_filters, dropout_prob=0.3) 
    cblock5 = conv_block(cblock4[0], 16 * n_filters, dropout_prob=0.3, max_pooling=False)     
    
    # Expanding path
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8 * n_filters)
    ublock7 = upsampling_block(ublock6, cblock3[1], 4 * n_filters)
    ublock8 = upsampling_block(ublock7, cblock2[1], 2 * n_filters)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(
        n_filters, 3, activation='relu', padding='same',
        kernel_initializer='he_normal'
    )(ublock9)
    
    # Final layer generates `n_classes` logits 
    # (used with SparseCategoricalCrossentropy(from_logits=True))
    outputs = Conv2D(n_classes, kernel_size=1, padding='same')(conv9)  
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
