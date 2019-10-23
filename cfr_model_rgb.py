import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

#loading features and labels
X_rgb = pickle.load(open("X_rgb.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#normalizing values, also to convert into float data
X_rgb = X_rgb/255.0

#creating the model
model = Sequential()


model.add(Conv2D(64, (3, 3), input_shape=X_rgb.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.compat.v1.keras.layers.Dropout(rate=.15))


model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.compat.v1.keras.layers.Dropout(rate=.15))


model.add(Flatten())
model.add(Dense(64))


model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

#fitting model, it will be evaluated on 20% of the whole data
model.fit(X_rgb, y, batch_size=32, epochs=10, validation_split=0.2)

#saving the model
model.save('dog_cat_cfr_rgb.model')

