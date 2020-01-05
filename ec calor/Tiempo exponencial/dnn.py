from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

outFile = "heat_eqs.eqs"
test_size = 0.2

def main():

	with open(outFile, "rb") as file:
		mat = np.load(file, allow_pickle=True)

	X = [m[0] for m in mat]
	y = [m[1] for m in mat]
	X = np.array(X)
	y = np.array(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

	model = Sequential()
	model.add(Dense(100, input_dim=100, activation='linear'))
	model.add(Dense(1250, activation='linear'))
	model.add(Dense(2500, activation='linear'))
	model.add(Dense(5000, activation='linear'))
	model.add(Dense(5000, activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y_train, epochs=20, batch_size=100)

	_, accuracy = model.evaluate(X_test, y_test)
	print('Accuracy: %.2f' % (accuracy*100))

	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")

if __name__ == '__main__':
	main()
