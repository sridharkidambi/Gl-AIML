import tensorflow;

print(tensorflow.__version__)
import random;
random.seed(0)
import warnings;
warnings.filterwarnings("ignore");

from tensorflow.keras.datasets import boston_housing;

(features,actual_prices), _=boston_housing.load_data(test_split=0)

# print(features[0][0][0])
print(actual_prices)
print(features.shape[0])
print(features.shape[1])
print(actual_prices.shape)
print(features[:5])
print(actual_prices[:5])

# define the model
model =tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.BatchNormalization(input_shape=(13,)))
model.add( tensorflow.keras.layers.Dense(1) )

model.compile(optimizer="sgd",loss="mse")
model.fit(features,actual_prices,epochs=124,validation_split=0.35)

import numpy as np
test_x = np.reshape([1.2, 0, 8.14e+00, 0.0e+00, 5.3e-01, 6.14e+00, 9.170e+01, 3.97e+00, 4, 3.07e+02, 2.10e+01, 3.96e+02, 1.872e+01],(-1, 13))
test_y = model.predict(test_x)
print(test_y)

