import tensorflow
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers, optimizers
import math

random.seed(0)

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")



# the data, shuffled and split between train and test sets
(X_train, y_train), (X_val, y_val) = mnist.load_data()

print("Label: {}".format(y_train[0]))
plt.imshow(X_train[0], cmap='gray')

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
# plt.show()
X_train = X_train.reshape(60000, 784)
print(X_train.shape)
X_val = X_val.reshape(10000, 784)
print(X_val.shape)

print(X_train.max())
print(X_train.min())

X_train = X_train / 255.0
X_val = X_val / 255.0

print(X_train.max())
print(X_train.min())

print(y_train[10])
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes=10)
print(y_train[10])

plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.axis('off')
    print('label for each of the below image: %s' % (np.argmax(y_train[0:10][i])))
# plt.show()

### Creating model 1

def train_and_test_loop(iterations, lr, Lambda, verb=True):
    
    ## hyperparameters
    iterations = iterations
    learning_rate = lr
    hidden_nodes = 256
    output_nodes = 10
        
    model = Sequential()
    model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))
    model.add(Dense(hidden_nodes, activation='relu'))
    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
    
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Fit the model
    model.fit(X_train, y_train, epochs=iterations, batch_size=1000, verbose= 1)


### Creating model 2
# - Same model as above
# - Instead of accuracy at each epoch below code gives the consolidate accuracy
# - Notice: The model.evaluate line at the last is the only difference from model 1

def train_and_test_loop1(iterations, lr, Lambda, verb=True):
    
    ## hyperparameters
    iterations = iterations
    learning_rate = lr
    hidden_nodes = 256
    output_nodes = 10

    model = Sequential()
    model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))
    model.add(Dense(hidden_nodes, activation='relu'))
    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
    
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Fit the model
    model.fit(X_train, y_train, epochs=iterations, batch_size=1000, verbose= 1)
    score = model.evaluate(X_train, y_train, verbose=0)
    
    return score

lr = 0.00001
Lambda = 0
train_and_test_loop(1, lr, Lambda)

lr = 0.00001
Lambda = 1e3
train_and_test_loop(1, lr, Lambda)

### Now, lets overfit to a small subset of our dataset, in this case 20 images, to ensure our model architecture is good

X_train_subset = X_train[0:20]
y_train_subset = y_train[0:20]
X_train = X_train_subset
y_train = y_train_subset
X_train.shape
y_train.shape

lr = 0.001
Lambda = 0
train_and_test_loop(500, lr, Lambda)
### Very small loss,  train accuracy going to 100, nice! We are successful in overfitting. The model architecture looks fine. Lets go for fine tuning it.

(X_train, y_train), (X_val, y_val) = tensorflow.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000, 784)
print(X_train.shape)
X_val = X_val.reshape(10000, 784)
print(X_val.shape)

X_train = X_train / 255.0
X_val = X_val / 255.0

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes=10)

lr = 1e-7
Lambda = 1e-7
train_and_test_loop(20, lr, Lambda)

### Loss barely changing. Learning rate is probably too low.
### Okay now lets try a (larger) learning rate 1e6. What could possibly go wrong?

### Okay now lets try a (larger) learning rate 1e6. What could possibly go wrong?

# - Learning rate lr = 1e8
# - Regularization lambda = 1e-7

lr = 1e8
Lambda = 1e-7
train_and_test_loop(20, lr, Lambda)

for k in range(1,10):
    lr = math.pow(10, np.random.uniform(-7.0, 3.0))
    Lambda = math.pow(10, np.random.uniform(-7,-2))
    best_acc = train_and_test_loop1(100, lr, Lambda, False)
    print("Try {0}/{1}: Best_val_acc: {2}, lr: {3}, Lambda: {4}\n".format(k, 100, best_acc, lr, Lambda))

### As you can see from above, Case 2, 3 and 7 yields good accuracy. It is better to focus on those values for learning rate and Lambda
for k in range(1,5):
    lr = math.pow(10, np.random.uniform(-4.0, -1.0))
    Lambda = math.pow(10, np.random.uniform(-4,-2))
    best_acc = train_and_test_loop1(100, lr, Lambda, False)
    print("Try {0}/{1}: Best_val_acc: {2}, lr: {3}, Lambda: {4}\n".format(k, 100, best_acc, lr, Lambda))

lr = 2e-2
Lambda = 1e-4
train_and_test_loop1(100, lr, Lambda)