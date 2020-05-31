import matplotlib
import numpy as np
import keras

from matplotlib import pyplot as plt

from keras.layers import Input, Dense, Softmax
from keras.models import Model

from keras.utils import to_categorical

from keras.models import Sequential

from keras.utils import plot_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# %matplotlib inline


dataset = np.loadtxt("somedata.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,1:3]
Y = dataset[:,0]
X1 = dataset[:,1:2]
X2 = dataset[:,2:3]

print (X)
print (Y)

color= ['red' if l == -1 else 'green' for l in Y]
plt.scatter(X1, X2, color=color)
plt.show()


X = np.array(X)
y = np.array(Y)

print("X shape:",X.shape)
print("y shape:",y.shape)

y_probs = to_categorical(y, num_classes=2)
print("New y sample:")
print(y_probs[:10, :])



model = Sequential()
model.add(Dense(2, input_shape=(2,)))
model.add(Softmax())

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=['acc'])
model.summary()


history = model.fit(X, y_probs, validation_split=0.33, epochs=10, batch_size=10, verbose=0)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


w, b = model.layers[0].get_weights()

print (w)

print (b)


SVG(model_to_dot(model).create(prog='dot', format='svg'))