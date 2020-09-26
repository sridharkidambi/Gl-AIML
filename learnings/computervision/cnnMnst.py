import tensorflow
import random
random.seed(0)
import warnings
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers, optimizers
import math
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.utils import to_categorical


(X_train,y_train), (X_test,y_test)=mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# y_val = to_categorical(y_val, num_classes=10)

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',metrics=["accuracy"],optimizer='adam')
model.fit(x=X_train,y=y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))