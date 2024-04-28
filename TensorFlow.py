import tensorflow as tf; tf.keras
from tensorflow.keras import layers, models, losses
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape

x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
x_train.shape

x_train = tf.expand_dims(x_train, axis = 3, name = None)
x_test = tf.expand_dims(x_test, axis = 3, name = None)
x_train.shape

x_val = x_train[-2000:, :, :, :]
y_val = y_train[-2000:]
x_train = x_train[:-2000, :, :, :]
y_train = y_train[:-2000]

model = models.Sequential([
    layers.Conv2D(6, 5, activation = 'tanh', input_shape=x_train.shape[1:]),
    layers.AveragePooling2D(2),
    layers.Activation('sigmoid'),
    layers.Conv2D(16, 5, activation = 'tanh'),
    layers.AveragePooling2D(2),
    layers.Activation('sigmoid'),
    layers.Conv2D(120, 5, activation = 'tanh'),
    layers.Flatten(),
    layers.Dense(84, activation = 'tanh'),
    layers.Dense(10, activation = 'softmax')
])
model.summary()

model.compile(optimizer = 'adam', loss = losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_data = (x_val, y_val))

model.evaluate(x_test, y_test)