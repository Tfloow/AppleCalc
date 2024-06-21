import numpy as np
import NN as NN
import glob
import matplotlib.image as mpim
# Parse command
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Train the neural network")
parser.add_argument("--Training", help="Load NPY data", action="store_true")
parser.add_argument("--Weight", help="Use Saved weights", action="store_true")
args = parser.parse_args()

network = NN.NeuralNetwork(28*28, 200, 10, 0.3)

# Check if the training data is already saved
if args.Training:
    print("[LOG] : Loading training data")
    training_data = np.load("training_data_image.npy")
    targets = np.load("training_data_target.npy")
    
    training_data = [(image, target) for image, target in zip(training_data, targets)]
    
else:
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
        
        
# Save the training data
np.save("training_data_image.npy", np.array([image for image, _ in training_data]))
np.save("training_data_target.npy", np.array([target for _, target in training_data]))
    
# Train the network and shuffle the data
if args.Weight:
    print("[LOG] : Loading weights")
    network.loadWeight()
else:
    print("[LOG] : Training the network")

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

# Backtrack the network

for i in range(10):
    target = np.zeros((10, 1)) + 0.01
    target[i] = 0.99
    inputs = network.backquery(target)

    
