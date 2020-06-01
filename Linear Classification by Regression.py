import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation, rc

# %matplotlib inline

rc('animation', html='html5')


fig, ax = plt.subplots();

ax.set_xlim([-15,15])
ax.set_ylim([-6,6])
line1, = ax.plot([], [], '-')
line2 = ax.scatter([], [])

plt.show()

data = [(2,0),(5,-2),(-2,2),(-1,-3)]
labels = [-1,-1,1,1]

X = np.array(data)
y = np.array(labels)

print("X shape:",X.shape)
print("y shape:",y.shape)

model = Sequential()
model.add(Dense(1, input_shape=(2,)))

model.compile(loss="mse", optimizer="sgd", metrics=['acc'])
model.summary()

parameter_history = []
for epoch in range(50):
    # Perform one step over the entire dataset
    loss_history = model.fit(X, y, epochs=1, verbose=False)

    # Get predictions (value of the objective function, f)
    y_pred = model.predict(X, verbose=False)

    # See how well our model is doing
    # Recall our classes are [-1,-1,1,1]
    num_correct = 0
    if y_pred[0] < 0: num_correct += 1
    if y_pred[1] < 0: num_correct += 1
    if y_pred[2] > 0: num_correct += 1
    if y_pred[3] > 0: num_correct += 1
    acc = num_correct / 4.0
    loss = loss_history.history['loss'][-1]
    print("Epoch %d: %0.2f (acc) %0.2f (loss)" % (epoch + 1, acc, loss))

    # Not mandatory: Save parameters for later analysis
    w, b = model.layers[0].get_weights()
    parameter_history.append((w, b))

plt.scatter([x[0] for x in data],
            [x[1] for x in data],
            c=['b', 'b', 'r', 'r'],
            s=40)

x1 = np.arange(-20, 20, 0.1)
x2 = (-1 * b - (w[0] * x1)) / w[1]
plt.axis([-15, 15, -6, 6])
plt.plot(x1, x2)

plt.show()

def show_learning(history):
    global iteration
    global fig
    iteration = 0
    # initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        iteration = 0
        return (line1,line2,)

    # animation function. This is called sequentially
    def animate(i):
        global iteration
        iteration = min(iteration, len(history)-1)
        x = np.arange(-20,20,0.1)
        y = (-1 * history[iteration][1] - (history[iteration][0][0] * x)) / history[iteration][0][1]

        line1.set_data(x, y)
        line2 =  ax.scatter([p[0] for p in data], [p[1] for p in data], c=['b','b','r','r'], s=40)

        iteration += 1

        return (line1,line2)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(history)+5, interval=250, blit=True)

    return anim
show_learning(parameter_history)


