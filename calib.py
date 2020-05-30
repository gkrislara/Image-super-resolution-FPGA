from PIL import Image
import os
import numpy as np 
import glob
import random


calib_image_dir = "path/to/image/directory"

calib_batch_size = 50

image_files = [f for f in glob.glob(calib_image_dir+'*.png')]


random_index=[]
for i in range(100):
    random_index.append(random.randrange(0,len(image_files)))
random_index=np.asarray(random_index)

def calib_input(iter):
    images=[]
    for index in range(0, calib_batch_size):
        curimg = random_index[iter * calib_batch_size + index]
        filename = image_files[curimg]
        im = Image.open(filename)
        image = np.asarray(im)
        image=image/255.0
        images.append(image.tolist())
    return {"LR": images}
