from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)

	# load pima indians dataset
	dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
	# split into input (X) and output (Y) variables
	X = dataset[:,0:8]
	Y = dataset[:,8]

	# split data into training, test set
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# Fit the model
	model.fit(X_train, y_train, nb_epoch=150, batch_size=10, verbose=0)

	# evaluate the model
	scores = model.evaluate(X_train, y_train)
	print()
	print("Training %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	test_scores = model.evaluate(X_test, y_test)
	print()
	print("Test %s: %.2f%%" % (model.metrics_names[1], test_scores[1]*100))


	print('\n\nUsing scikit-learn to train model')
	mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 8), 
		                max_iter = 1000, random_state=1, early_stopping=False)
	mlp.fit(X_train, y_train)

	y_train_pred = mlp.predict(X_train)
	print('Training acc: %.2f' % (accuracy_score(y_train, y_train_pred)))

	y_test_pred = mlp.predict(X_test)
	print('Test acc: %.2f' % (accuracy_score(y_test, y_test_pred)))

	# calculate predictions
	# predictions = model.predict(X)
	# round predictions
	# rounded = [x[0].round() for x in predictions]
	# print(rounded)

if __name__ == '__main__':
	main()