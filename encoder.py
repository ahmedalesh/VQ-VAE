import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim=10, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))

        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')
        self.dropout3 = tf.keras.layers.Dropout(0.25)
        self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv4 = tf.keras.layers.Conv2D(filters=embedding_dim, kernel_size=(1,1), padding='same')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.maxpool3(x)

        return self.conv4(x)

    def get_config(self):
        new_config = {'embedding_dim': self.embedding_dim}
        config = super(EncoderLayer, self).get_config()
        return dict(list(config.items()) + list(new_config.items()))
