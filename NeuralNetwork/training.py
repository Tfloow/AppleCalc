import numpy as np
import NN as NN
from scipy import misc
import glob
import matplotlib.image as mpim

network = NN.NeuralNetwork(28*28, 200, 10, 0.3)

# open and load the mnist training data PNG file
training_data = []
labels = []
for number in range(10):
    for i, image_path in enumerate(glob.glob(f"mnist_png/training/{number}/*.png")):
        image = mpim.imread(image_path)
        image = np.reshape(image, (28*28, 1))
        image = image / 255.0 * 0.99 + 0.01
        target = np.zeros((10, 1)) + 0.01
        target[number] = 0.99
        training_data.append((image, target))
        labels.append(number)
        
        if i > 100:
            break
        
# Train the network and shuffle the data

np.random.shuffle(training_data)
for image, target in training_data:
    network.train(image, target)
        
network.writeWeight()

# Test the network
test_data = []
test_labels = []
for number in range(10):
    for i, image_path in enumerate(glob.glob(f"mnist_png/testing/{number}/*.png")):
        image = mpim.imread(image_path)
        image = np.reshape(image, (28*28, 1))
        image = image / 255.0 * 0.99 + 0.01
        target = np.zeros((10, 1)) + 0.01
        target[number] = 0.99
        test_data.append((image, target))
        labels.append(number)
        
        if i > 10:
            break
        
# Test the network
correct = 0
test = 0
for image, target in test_data:
    output = network.query(image)
    test += 1
    print(f"Output: {np.argmax(output)} Target: {np.argmax(target)}")
    if np.argmax(output) == np.argmax(target):
        correct += 1
        
print(correct/test)
        
