import NeuralNetwork.NN as NN
import argparse
import matplotlib.image as mpim
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np

# Create the parser
parser = argparse.ArgumentParser(description="Upload a picture and predict the number")
parser.add_argument("--Path", help="Path to the image", required=True)

args = parser.parse_args()

print("[LOG] : Creating the Neural Network")
network = NN.NeuralNetwork(28*28, 200, 10, 0.3)
network.loadWeight()
print("[LOG] : Weights loaded")

# Read the image and predict the number
image = mpim.imread(args.Path)
image = np.array(image)
# convert to greyscale and invert the image
if len(image.shape) > 2:
    image = image[:,:,0]
    image = 1.0 - np.array(image)
if image.shape[0] != image.shape[1] :
    # Pad the image
    padding_rows = max(0, image.shape[1] - image.shape[0])
    padding_cols = max(0, image.shape[0] - image.shape[1])
    image = np.pad(image, ((0, padding_rows), (0, padding_cols)), mode='constant')
    print(image.shape)
# Need to enhance the cropping with image detection algorithm
if image.shape[0] != 28:
    image = zoom(image, 28/image.shape[0])
    print(image.shape)
    
image = image / 255.0 * 0.99 + 0.01


plt.imshow(image)
plt.show()

print(network.guess(image))