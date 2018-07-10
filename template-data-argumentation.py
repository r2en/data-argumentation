import os
import glob
#import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

IMAGE_FILE = "cat.jpg"

def draw_image(data_argumentation, numpy_array, \
               result_images, temporary_directory, number):

    # generate 9 images by ImageDataGenerator
    generate_data = data_argumentation.flow(numpy_array, batch_size=1,\
                            save_to_dir = temporary_directory,\
                            save_prefix='image', save_format='jpg')

    for i in range(9):
        batch = generate_data.next()

if __name__ == '__main__':

    # preprocessing image
    image = load_img(IMAGE_FILE)
    numpy_array = img_to_array(image)
    numpy_array = np.expand_dims(numpy_array, axis=0)

    temporary_directory = "Data-Argument"
    os.mkdir(temporary_directory)

    # load ImageDataGenerator & processing images
    for number in range(90):
        data_argumentation = ImageDataGenerator(rotation_range=number)
        draw_image(data_argumentation, numpy_array, \
                "result_rotation90.jpg", temporary_directory, number)