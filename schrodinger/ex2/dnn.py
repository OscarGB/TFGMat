import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as mp
import tensorflow as tf
import keras


def plot(trainx, trainy, model):
    mp.plot(model.predict(np.array([trainx]))[0])
    mp.plot([trainx[i]/max(trainx) for i in range(bins - 1)])
    mp.plot(trainy)
    mp.show()
    

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

bins = 128
seedmax = 60 # opens seed files 0 - 19. Lost too much data due to kernel crashes, so these got broken up
trainx = []
trainy = []
validx = []
validy = []

#This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(seedmax):
    with open('test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open('test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open('valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open('valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])
            
model = Sequential()
model.add(Dense(127, input_dim=127, activation='softplus'))
model.add(Dense(127, activation='softplus'))
model.add(Dense(127, activation='softplus'))
model.add(Dense(127, activation='softplus'))
model.add(Dense(127, activation='softplus'))
model.add(Dense(127, activation='softplus'))
model.compile(loss='mean_squared_error', optimizer='adam')

trainx = np.array(trainx)
trainy = np.array(trainy)
validx = np.array(validx)
validy = np.array(validy)

print(validy)
print(validx)

model.fit(trainx, trainy, epochs=100000, batch_size=960, verbose=2)
accuracy = model.evaluate(validx, validy)
print('Accuracy: %.2f' % (accuracy*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
