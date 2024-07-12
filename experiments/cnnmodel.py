import keras
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

class ModelSubClassing(keras.Model):
    def __init__(self, num_classes, l1_reg=0.01, l2_reg=0.001, alpha = 0.001):
        super().__init__()
        
        # define L1 and L2 regularizer
        self.l1_l2_regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        # self.l2_regularizer = regularizers.l2(l2_reg)
        
        # define alpha the loss regulator
        self.alpha = alpha
        
        # number of classes
        self.num_classes = num_classes
        
        # initialize layers as None
        self.conv1 = None
        self.max1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.max2 = None
        self.drop1 = None
        self.conv3 = None
        self.bn3 = None
        self.max3 = None
        self.drop2 = None
        self.flatten = None
        self.dense128 = None
        self.dense128_activation = None
        self.dense128_do = None
        self.dense64 = None
        self.dense64_activation = None
        self.dense_out = None
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = keras.metrics.CategoricalAccuracy(name="accuracy")

    def build(self, input_shape):
        # create layers
        self.conv1 = Conv2D(32, 3, activation="relu", kernel_regularizer=self.l1_l2_regularizer)
        self.max1  = MaxPooling2D(3)
        self.bn1   = BatchNormalization(momentum=0.9)

        self.conv2 = Conv2D(64, 3, activation="relu", kernel_regularizer=self.l1_l2_regularizer)
        self.bn2   = BatchNormalization(momentum=0.9)
        self.max2  = MaxPooling2D(3)
        self.drop1 = Dropout(0.3)

        self.conv3 = Conv2D(128, 3, activation="relu", kernel_regularizer=self.l1_l2_regularizer)
        self.bn3   = BatchNormalization(momentum=0.9)
        self.max3  = MaxPooling2D(3)
        self.drop2 = Dropout(0.3)

        self.flatten = Flatten()
        self.dense128 = Dense(128, kernel_regularizer=self.l1_l2_regularizer)
        self.dense128_activation = Activation('relu')
        self.dense128_do = Dropout(0.3)
        self.dense64 = Dense(64, kernel_regularizer=self.l1_l2_regularizer)
        self.dense64_activation = Activation('relu')
        self.dense_out = Dense(self.num_classes, activation="softmax")
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Unpack the inputs
        mel_spectrogram = inputs[0]

        # Forward pass: block 1
        x = self.conv1(mel_spectrogram)
        x = self.max1(x)
        x = self.bn1(x, training=training)

        # Forward pass: block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.max2(x)
        x = self.drop1(x, training=training)

        # Forward pass: block 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.max3(x)
        x = self.drop2(x, training=training)

        # Flatten and dense layers
        x = self.flatten(x)
        x_dense128 = self.dense128(x)
        x_dense128_act = self.dense128_activation(x_dense128)
        if training:
            x_dense128_act = self.dense128_do(x_dense128_act)
        x_dense64 = self.dense64(x_dense128_act)
        x_dense64_act = self.dense64_activation(x_dense64)
        y_hat = self.dense_out(x_dense64_act)
        
        return y_hat, x_dense64


    def train_step(self, data):
        # Unpack the data
        (mel_spectrogram, caption_embedding), y = data

        with tf.GradientTape() as tape: 
            y_pred, x_dense64 = self((mel_spectrogram, caption_embedding), training=True)
            # Compute three loss value
            loss = self.compiled_loss(y, y_pred)
            custom_loss = self.custom_loss(caption_embedding, x_dense64)
            total_loss = (1 - self.alpha) * loss + self.alpha * custom_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the loss metric 
        self.loss_tracker.update_state(total_loss)
        self.accuracy.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy.result(),
            "crossentropy_loss": loss,
            "embedding_loss": custom_loss,
            "total_loss": total_loss
        }

        
    def test_step(self, data):
        # unpack the data
        (mel_spectrogram, caption_embedding), y = data
        y_pred, x_dense64 = self((mel_spectrogram, caption_embedding), training=False)  #keep this training at false
        # compute three loss value
        loss = self.compiled_loss(y, y_pred)
        custom_loss = self.custom_loss(caption_embedding, x_dense64)
        total_loss = (1 - self.alpha) * loss + self.alpha * custom_loss

        # Update the loss metric and accuracy metrics
        self.loss_tracker.update_state(total_loss)
        self.accuracy.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy.result(),
            "crossentropy_loss": loss,
            "embedding_loss": custom_loss,
            "total_loss": total_loss
        }
        
    # add a custom loss method in the model
    def custom_loss(self, caption_embedding, x_dense64):
        return tf.reduce_mean(tf.square(caption_embedding - x_dense64))
    
    @property
    def metrics(self):
        # list all metrics to be reset after each epoch
        return [self.loss_tracker, self.accuracy]