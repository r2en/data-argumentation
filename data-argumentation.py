import os
import glob
#import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

IMAGE_FILE = "cat.jpg"

def draw_image(data_argumentation, numpy_array, result_images):
    temporary_directory = "Data-Argument"
    os.mkdir(temporary_directory)

    # generate 9 images by ImageDataGenerator
    generate_data = data_argumentation.flow(numpy_array, batch_size=1,\
                            save_to_dir = temporary_directory,\
                            save_prefix='img', save_format='jpg')

    for i in range(9):
        batch = generate_data.next()

    # generate 3 * 3 images by matplotlib's grid     
    images = glob.glob(os.path.join(temporary_directory, "*.jpg"))
    figure = plt.figure()
    gridspec_image = gridspec.GridSpec(3, 3)
    gridspec_image.update(wspace=0.1, hspace=0.1)

    for i in range(9):
        image = load_img(images[i])
        plt.subplot(gridspec_image[i])
        plt.imshow(image, aspect='auto')
        plt.axis("off")
    plt.savefig(result_images)

    #shutil.rmtree(temporary_directory)

if __name__ == '__main__':

    # preprocessing image
    image = load_img(IMAGE_FILE)
    numpy_array = img_to_array(image)
    numpy_array = np.expand_dims(numpy_array, axis=0)

    # load ImageDataGenerator & processing images
    data_argumentation = ImageDataGenerator(rotation_range=90)
    draw_image(data_argumentation, numpy_array, "result_rotation.jpg")