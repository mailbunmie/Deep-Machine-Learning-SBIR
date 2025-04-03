!pip install DeepImageSearch --upgrade

from google.colab import files
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# unit test on patching an image

# read image and resize to 128
#image = Image.open('imgs/4a.jpg').resize((128, 128))
#image = Image.open('/content/drive/MyDrive/21a.jpg').resize((128, 128))
image = Image.open('/content/drive/Othercomputers/8a.jpg').resize((128, 128))


# convert to numpy array
x = np.array(image)

# An Image Is Worth 16x16 Words
P = 16   # patch size
C = 3    # number of channels (RGB)

# split image into patches using numpy
patches = x.reshape(x.shape[0]//P, P, x.shape[1]//P, P, C).swapaxes(1, 2).reshape(-1, P, P, C)

# flatten patches
x_p = np.reshape(patches, (-1, P * P * C))

# get number of patches
N = x_p.shape[0]

print('Image shape: ', x.shape)  # width, height, channel
print('Number of patches: {} with resolution ({}, {})'.format(N, P, P))
print('Patches shape: ', patches.shape)
print('Flattened patches shape: ', x_p.shape)

# visualize data
#
# display image and patches side-by-side

fig = plt.figure()

gridspec = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gridspec[0])
ax1.set(title='Image')

# display image
ax1.imshow(x)

subgridspec = gridspec[1].subgridspec(8, 8, hspace=-0.8)

# display patches
for i in range(8):    # N = 64, 8x8 grid
    for j in range(8):
        num = i * 8 + j
        ax = fig.add_subplot(subgridspec[i, j])
        ax.set(xticks=[], yticks=[])
        ax.imshow(patches[num])