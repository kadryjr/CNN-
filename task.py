# Simple CNN for the MNIST Dataset
import keras.optimizers
from keras.datasets import mnist
from keras.models import Sequential  # Allows to build the architecture for neural network
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape(60000, 28, 28, 1)  # it makes images in grayscale
X_test = X_test.reshape(10000, 28, 28, 1)


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)  # it makes label 5 = [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define a simple CNN model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'))   # convolution layer to extract features from the input image
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())        # take the images and flatten them (turn images into a one dimensional array)
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    optimizer = keras.optimizers.SGD(lr=0.009)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# build the model
model = baseline_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, shuffle=True)        # epochs : number of iterations when an entire data set is passed forward and backward through the neural network
model.summary()
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
