import tensorflow as tf

class VQVAE(tf.keras.layers.Layer):
    def __init__(self, embedding_dim=10, num_embdedding=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embedding = num_embdedding

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name='embedding_weight', shape=(input_shape[-1], self.num_embedding),
            dtype=tf.float32, initializer='uniform')
        super(VQVAE, self).build(input_shape)

    def get_config(self):
        config = super(VQVAE, self).get_config()
        new_config = {'embedding_dim': self.embedding_dim, 'num_embedding':self.num_embedding}
        return dict(list(config.items()) + list(new_config.items()))

    def call(self, inputs, **kwargs):
        x = tf.keras.backend.reshape(inputs, (-1, self.embedding_dim))
        distance = tf.keras.backend.sum(tf.pow(x, 2), axis=-1, keepdims=True) - (2 * tf.keras.backend.dot(x, self.embedding)) + tf.keras.backend.sum(tf.pow(self.embedding, 2), axis=0, keepdims=True)
        arg_distance = tf.keras.backend.reshape(tf.keras.backend.argmax(-distance, axis=1), tf.keras.backend.shape(inputs)[:-1])
        return tf.nn.embedding_lookup(tf.keras.backend.transpose(self.embedding.read_value()), arg_distance)