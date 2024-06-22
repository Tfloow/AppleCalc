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
    
# Recenter the image
print(image)

for i in range(image.shape[0]):
    if np.max(image[i]) != 0:
        top = i
        break
for i in range(image.shape[0]-1, 0, -1):
    if np.max(image[i]) != 0:
        bottom = i
        break
for i in range(image.shape[1]):
    if np.max(image[:,i]) != 0:
        left = i
        break
for i in range(image.shape[1]-1, 0, -1):
    if np.max(image[:,i]) != 0:
        right = i
        break

border = 5
image = image[max(0,top-border):min(image.shape[0],bottom+border), max(0,left-border):min(image.shape[1], right+border)]

print(image.shape)

plt.imshow(image)
plt.show()
    
if image.shape[0] != image.shape[1] :
    # Pad the image
    padding_rows = max(0, image.shape[1] - image.shape[0])
    padding_cols = max(0, image.shape[0] - image.shape[1])
    image = np.pad(image, ((int(np.floor(padding_rows/2)), int(np.ceil(padding_rows/2)))
                           , (int(np.floor(padding_cols/2)), int(np.ceil(padding_cols/2)))), mode='constant')
    print(image.shape)
# Need to enhance the cropping with image detection algorithm
if image.shape[0] != 28:
    image = zoom(image, 28/image.shape[0])
    print(image.shape)
    
image = image / 255.0 * 0.99 + 0.01


plt.imshow(image)
plt.show()

print(network.guess(image))