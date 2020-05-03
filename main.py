import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from encoder import EncoderLayer
from decoder import DecoderLayer
from vqvae import VQVAE

def vqvae_loss(commitment, encoder, quantize):
    def vqvae_innerloss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        encoder_loss = tf.keras.backend.mean(tf.pow(tf.stop_gradient(quantize) - encoder,2))
        commitment_loss = tf.keras.backend.mean(tf.pow(quantize - tf.stop_gradient(encoder),2))
        return reconstruction_loss + encoder_loss + commitment* commitment_loss
    return vqvae_innerloss

def train_model(train_data, validation_data):
    model = construct_model()
    print(model.summary())
    history = model.fit(train_data, train_data, batch_size=32, epochs=100, validation_data=(validation_data, validation_data),
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint('model.h5',save_best_only=True, save_weights_only=True),
                  tf.keras.callbacks.EarlyStopping(patience=10),
                  tf.keras.callbacks.TensorBoard(log_dir='../', histogram_freq=5, write_graph=True,
                                                 write_images=True)
              ])
    model.save_weights('model.h5')
    loss = history.history['loss']  # Training loss.
    val_loss = history.history['val_loss']  # Validation loss.
    num_epochs = range(1, 1 + len(history.history['loss']))  # Number of training epochs.
    plt.figure(figsize=(16, 9))
    plt.plot(num_epochs, loss, label='Training loss')  # Plot training loss.
    plt.plot(num_epochs, val_loss, label='Validation loss')  # Plot validation loss.
    plt.title('Training and validation loss')
    plt.legend(loc='best')
    plt.show()


def predict(test_data):
    model = construct_model()
    model.load_weights('model.h5')
    prediction = model.predict(test_data)
    # Show original reconstruction.
    n_rows = 5
    n_cols = 8
    samples_per_col = int(n_cols / 2)
    sample_offset = np.random.randint(0, len(test_data) - n_rows - n_cols - 1)
    # sample_offset = 0
    img_idx = 0
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1, 2):
            idx = n_cols * (i - 1) + j
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.imshow(test_data[img_idx + sample_offset].reshape(28, 28),
                      cmap='gray_r',
                      clim=(0, 1))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction.
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            ax.title.set_text('({:d}) Reconstruction'.format(img_idx))
            ax.imshow(prediction[img_idx + sample_offset].reshape(28, 28), cmap='gray_r', clim=(0, 1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img_idx += 1
    plt.savefig('image_result.png')
    plt.show()
    plt.close()

def construct_model():
    x_input = tf.keras.layers.Input((28,28,1))
    enc_x = EncoderLayer()(x_input)
    quant_x = VQVAE()(enc_x)
    x_dec = tf.keras.layers.Lambda(lambda quant_x: enc_x + tf.stop_gradient(quant_x - enc_x))(quant_x)
    dec_x = DecoderLayer()(x_dec)
    model = tf.keras.models.Model(x_input, dec_x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=vqvae_loss(0.25, enc_x, quant_x), experimental_run_tf_function=False)
    return model

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train, x_valid = train_test_split(x_train, test_size=0.2, shuffle=True)
    train_model(x_train, x_valid)
    predict(x_test)

if __name__ == '__main__':
    main()






