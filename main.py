import NeuralNetwork.NN as NN

network = NN.NeuralNetwork(2, 1, [3, 3])

test = [1,2]
target = 1

network.train(test, target)