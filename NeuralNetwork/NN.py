import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt

"""
Il manque le dataset d'entrainement contenant 60 000 chiffres manuscrits car cela 
dépasse la taille maximale des fichiers sur github.
Téléchargeable ici: http://www.pjreddie.com/media/files/mnist_train.csv
"""

class NeuralNetwork:
    """
    My neural Network
    """
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set the number of node
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Set the learning rate
        self.lr = learningrate
        
        # Weights matrices
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        """
        Pour avoir la belle distribution des poids
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        """
        self.activation_function = lambda x: expit(x)
        self.inverse_activation_function = lambda x: logit(x)

    
    def train(self, inputs_list, targets_list):
        # Convert into a 2d array
        inputs = np.array(inputs_list, ndmin=2)
        targets = np.array(targets_list, ndmin=2)
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2)
             
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2)
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        mat = np.reshape(inputs, (28,28))
        plt.imshow(mat, interpolation='nearest', cmap=plt.cm.binary)
        plt.savefig(f"Back/Backtrack{np.argmax(targets_list)}")
        
        return inputs
    
    def writeWeight(self):
        np.savetxt("weightHidden.csv", self.wih, delimiter=",")
        
        np.savetxt("weightOutput.csv", self.who, delimiter=",")
        
    def loadWeight(self):
        self.wih = np.loadtxt("weightHidden.csv",delimiter=",", dtype=float)
        
        self.who = np.loadtxt("weightOutput.csv",delimiter=",", dtype=float)

    
class NeuralNetworkMultiple:
    """
    My neural Network
    """
    
    def __init__(self, layer_nodes : list[list[int]], learningrate : int):
        # Set the number of node
        self.nodes = layer_nodes
        
        # Set the learning rate
        self.lr = learningrate
        
        # Weights matrices
        self.weights = []
        for i in range(len(layer_nodes) - 1):
            self.weights.append(np.random.rand(self.nodes[i+1], self.nodes[i]) - 0.5)
        """
        Pour avoir la belle distribution des poids
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        """
        self.activation_function = lambda x: expit(x)
        self.inverse_activation_function = lambda x: logit(x)

    
    def train(self, inputs_list, targets_list):
        # Convert into a 2d array
        inputs = [np.array(inputs_list, ndmin=2)]
        outputs = []
        targets = np.array(targets_list, ndmin=2)
        
        for i in range(len(self.nodes) - 1):
            inputs.append(np.dot(self.weights[i], inputs[i]))
            outputs.append(self.activation_function(inputs[i+1]))
        
        errors = [outputs[-1] - targets] # First error then propagate
        for i in range(len(self.nodes) - 1):
            errors.append(np.dot(self.weights[len(self.nodes)-i-2].T, errors[i]))
                
        # Update the weights
        for i in range(1,len(self.nodes) - 1):
            self.weights[i] += self.lr * np.dot((errors[len(self.nodes)-i-2] * outputs[len(self.nodes)-i-1] * (1.0 - outputs[len(self.nodes)-i-1])), np.transpose(outputs[len(self.nodes)-i-2]))


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2)
        
        for i in range(len(self.nodes) - 1):
            inputs = np.dot(self.weights[i], inputs)
            inputs = self.activation_function(inputs)
        
        return inputs
    
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2)
        
        for i in range(len(self.nodes) - 1):
            final_inputs = self.inverse_activation_function(final_outputs)
            final_outputs = np.dot(self.weights[len(self.nodes)-i-2].T, final_inputs)
            final_outputs -= np.min(final_outputs)
            final_outputs /= np.max(final_outputs)
            final_outputs *= 0.98
            final_outputs += 0.01
        
        mat = np.reshape(final_outputs, (28,28))
        plt.imshow(mat, interpolation='nearest', cmap=plt.cm.binary)
        plt.savefig(f"Back/Backtrack{np.argmax(targets_list)}")
        
        return final_outputs
    
    def writeWeight(self):
        np.savetxt("weightHidden.csv", self.weights[0], delimiter=",")
        
        np.savetxt("weightOutput.csv", self.weights[1], delimiter=",")
        
    def loadWeight(self):
        self.weights = []
        self.weights.append(np.loadtxt("weightHidden.csv",delimiter=",", dtype=float))
        
        self.weights.append(np.loadtxt("weightOutput.csv",delimiter=",", dtype=float))

if __name__ == "__main__":
    input_nodes = 3; hidden_nodes = 3; output_nodes = 3
    learning_rate = 0.3
    
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    output = n.query([1.0,0.5,-1.5])
    print(output)