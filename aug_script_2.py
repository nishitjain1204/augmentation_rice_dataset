from keras.preprocessing.image import ImageDataGenerator
from skimage import io

import numpy as np
import os
from PIL import Image
# Using TensorFlow backend.
datagen = ImageDataGenerator(
        
#        brightness_range=None,
        shear_range=0.2,
        
        horizontal_flip=True,
        vertical_flip=True,
        
#        validation_split=0.0,
#        dtype=None,
        )
image_directory = "/home/nishit/augmentation/Rice_All"
SIZE = 256

my_folders = os.listdir(image_directory)

# print(my_images)

for folder in my_folders:
    os.makedirs('augmented_images'+'/'+folder)
    my_images = os.listdir(image_directory +'/'+folder)
    dataset = []
    for i, image_name in enumerate(my_images):
        # print(image_name)
        if ((image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'JPG')):
            image = io.imread(image_directory +'/'+folder+'/'+image_name)        
            image = Image.fromarray(image, 'RGB')        
            image = image.resize((SIZE,SIZE)) 
            dataset.append(np.array(image))
        
    x = np.array(dataset)
    print(len(x))
    # Generating Augmented Images
    batch_size = 20 # how many images should be created at a time
    num_of_count = 50
    num_of_images = batch_size * num_of_count

    count = 1
    for batch in datagen.flow(x, batch_size=batch_size, save_to_dir='augmented_images'+'/'+folder,
                            save_prefix='aug', save_format='jpg'):
        count += 1    
        if count > num_of_count:
            break
            
    print(f"{num_of_images } are generated for {folder}")