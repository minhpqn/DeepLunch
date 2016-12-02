CC := gcc
CFLAGS := -lm -O2 -funroll-loops  -std=c99
.PHONY: all target tests

all: logisticRegression neuralNetwork recurrentNeuralNetwork simpleRecurrentNeuralNetwork

logisticRegression: logisticRegression.c
	$(CC) logisticRegression.c -o logisticRegression $(CFLAGS)
neuralNetwork: neuralNetwork.c
	$(CC) neuralNetwork.c -o neuralNetwork $(CFLAGS)
recurrentNeuralNetwork: recurrentNeuralNetwork.c
	$(CC) recurrentNeuralNetwork.c -o recurrentNeuralNetwork $(CFLAGS)
simpleRecurrentNeuralNetwork: simpleRecurrentNeuralNetwork.c
	$(CC) simpleRecurrentNeuralNetwork.c -o simpleRecurrentNeuralNetwork $(CFLAGS)

clean:
	rm recurrentNeuralNetwork neuralNetwork logisticRegression simpleRecurrentNeuralNetwork

