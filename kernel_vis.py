import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn import *


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
first_layer_weights = conv_net.conv1.weight.data.cpu().numpy()

# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels,
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be
# between 0 and 1 before plotting.

num_kernels = first_layer_weights.shape[0]
num_cols = 8
num_rows = (num_kernels + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

for i in range(num_kernels):
    row = i // num_cols
    col = i % num_cols
    kernel = first_layer_weights[i, 0]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    axes[row, col].imshow(kernel, cmap='gray')
    axes[row, col].axis('off')

for i in range(num_kernels, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    fig.delaxes(axes[row, col])

# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.tight_layout()
plt.savefig('kernel_grid.png')
plt.close()

# Apply the kernel to the provided sample image.
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0               # Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image
output = F.conv2d(img, conv_net.conv1.weight, conv_net.conv1.bias, padding=1)
output = F.relu(output)  # Apply ReLU to match the network's activation

# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.
output = output.squeeze(0)
output = output.unsqueeze(1)

# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.
num_features = output.shape[0]
num_cols = 8
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
output_np = output.squeeze(1).detach().cpu().numpy()

for i in range(num_features):
    row = i // num_cols
    col = i % num_cols
    feature_map = output_np[i]
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    axes[row, col].imshow(feature_map, cmap='gray')
    axes[row, col].axis('off')

for i in range(num_features, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    fig.delaxes(axes[row, col])

# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.tight_layout()
plt.savefig('image_transform_grid.png')
plt.close()