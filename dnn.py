from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

outFile = "heat_eqs"
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
	model.add(Dense(256, input_dim=256, activation='relu'))
	for i in range(2,26):
		model.add(Dense(512*i, activation='relu'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y_train, epochs=50, batch_size=20)

	_, accuracy = model.evaluate(X_test, y_test)
	print('Accuracy: %.2f' % (accuracy*100))


if __name__ == '__main__':
	main()