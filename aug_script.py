import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn import metrics
import os


data_transforms = {
     transforms.Compose([
        transforms.Resize(256),
    #    transforms.hflip(),
    #    transforms.vflip(),
    #    transforms.affine(shear)
        
        
    ])
   
}

data_dir = "Rice_all"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['BrownSpot', 'Healthy','Hispa','LeafBlast']}

print(image_datasets)


# batch_size = 20 # how many images should be created at a time
# num_of_count = 50
# num_of_images = batch_size * num_of_count

# count = 1
# for batch in image_datasets.flow(x, batch_size=10, save_to_dir='/Users/sangkinam/mywork/w210/data/augmented_data/Raspberry___healthy',
#                           save_prefix='aug', save_format='jpg'):
#     count += 1    
#     if count > num_of_count:
#         break
        
# print("%d images are generated" % num_of_images)

