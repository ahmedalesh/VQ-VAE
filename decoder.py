import tensorflow as tf

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, n_channels=1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.upsample2 = tf.keras.layers.UpSampling2D(size=(2,2))

        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')
        self.dropout3 = tf.keras.layers.Dropout(0.25)
        self.upsample3 = tf.keras.layers.UpSampling2D(size=(2,2))

        self.conv4 = tf.keras.layers.Conv2D(filters=self.n_channels, kernel_size=(1,1), padding='same')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.upsample2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.upsample3(x)

        return self.conv4(x)

    def get_config(self):
        new_config = {'n_channels': self.n_channels}
        config = super(DecoderLayer, self).get_config()
        return dict(list(config.items()) + list(new_config.items()))