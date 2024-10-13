import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

def run_test():
    class CustomLayer(Layer):
        def __init__(self, units=2048):
            super(CustomLayer, self).__init__()
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer='random_normal',
                                     trainable=True)
            self.b = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    model = Sequential([
        Dense(2048, activation='relu', input_shape=(4096,)),
        Dropout(0.5),
        CustomLayer(2048),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        CustomLayer(1024),
        Dropout(0.5),
        Dense(512, activation='relu'),
        CustomLayer(512),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='mean_squared_error')

    x_train = np.random.random((100000, 4096)).astype(np.float32)
    y_train = np.random.random((100000, 1)).astype(np.float32)

    epochs = 1
    loss_object = tf.keras.losses.MeanSquaredError()
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(x_train, training=True)
            loss = loss_object(y_train, predictions)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print(f"Epoch {epoch + 1}, Loss: {np.mean(loss.numpy())}")


    # model.save('tests/my_model.h5')
    # print("Model saved!")

    # model.save_weights('tests/my_model_weights.h5')
    # print("Weights saved!")
    
    print("Training completed!")

