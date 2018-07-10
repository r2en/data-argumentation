# Data Argumentation(画像の水増し)


### TL;DR:
最近転移学習等の言葉も流行する通り、機械学習用のデータ(画像)を収集するのは大変でありデータの水増し、データの拡張を前処理として行うことは必要不可欠になりつつある。

DataArgumentationとは画像認識精度の向上を目的とし、訓練データの画像に対して、シフト、回転、拡大、縮小、色彩変化などを加えてロバストにすることである。

音声認識の分野でも人工的なノイズを加えてデータを水増しするテクニックが存在する。

### ImageDataGenerator
keras.preprocessing.imageにはrandom_rotationや、random_shiftなどの前処理が用意されているが、今回はより簡単に拡張ができるImageDataGeneratorクラスを紹介する

```python
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

IMAGE_FILE = "cat.jpg"

def draw_image(data_argumentation, numpy_array, result_images):
    temporary_directory = "tmp"
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

    # plot image
    for i in range(9):
        image = load_img(images[i])
        plt.subplot(gridspec_image[i])
        plt.imshow(image, aspect='auto')
        plt.axis("off")
    plt.savefig(result_images)

    # remove directory
    shutil.rmtree(temporary_directory)

if __name__ == '__main__':

    # preprocessing image
    image = load_img(IMAGE_FILE)
    numpy_array = img_to_array(image)
    numpy_array = np.expand_dims(numpy_array, axis=0)

    # load ImageDataGenerator & processing images
    data_argumentation = ImageDataGenerator(rotation_range=90)
    draw_image(data_argumentation, numpy_array, "result_rotation.jpg")
    
    data_argumentation = ImageDataGenerator(width_shift_range=0.2)
    draw_image(data_argumentation, numpy_array, "result_width_shift.jpg")

    data_argumentation = ImageDataGenerator(height_shift_range=0.2)
    draw_image(data_argumentation, numpy_array, "result_height_shift.jpg")
    
    data_argumentation = ImageDataGenerator(shear_range=0.78)
    draw_image(data_argumentation, numpy_array, "result_shear.jpg")

    data_argumentation = ImageDataGenerator(zoom_range=0.5)
    draw_image(data_argumentation, numpy_array, "result_zoom.jpg")
    
    data_argumentation = ImageDataGenerator(channel_shift_range=100)
    draw_image(data_argumentation, numpy_array, "result_channel_shift.jpg")

    data_argumentation = ImageDataGenerator(horizontal_flip=True)
    draw_image(data_argumentation, numpy_array, "result_horizontal_flip.jpg")

    data_argumentation = ImageDataGenerator(vertical_flip=True)
    draw_image(data_argumentation, numpy_array, "result_vertical_flip.jpg")

    data_argumentation = ImageDataGenerator(samplewise_center=True)
    draw_image(data_argumentation, numpy_array, "result_samplewise_center.jpg")

```

### 結果

指定角度の範囲でランダムに回転: rotation_range

![result_rotation90](https://user-images.githubusercontent.com/28590220/29109553-5ade0ede-7d1e-11e7-8666-09e2cd8a185d.jpg)

水平方向にランダムに移動: width_shift_range

![result_width_shift](https://user-images.githubusercontent.com/28590220/29112617-60d78738-7d29-11e7-88fa-0843f24d2c04.jpg)

垂直方向にランダムに移動: height_shift_range

![result_height_shift](https://user-images.githubusercontent.com/28590220/29112612-60aec820-7d29-11e7-8d9d-646b997ee154.jpg)

引き延ばすシアー変換をかける: shear_range

![result_shear](https://user-images.githubusercontent.com/28590220/29112613-60af9368-7d29-11e7-80d6-410f8d67f880.jpg)

ズームをランダムにする: zoom_range

![result_zoom](https://user-images.githubusercontent.com/28590220/29112616-60d435e2-7d29-11e7-9ee4-91cb4622a6cf.jpg)

チャンネルをランダムに移動: channel_shift_range

![result_channel_shift](https://user-images.githubusercontent.com/28590220/29112614-60c83f30-7d29-11e7-90ef-72a173a8ee39.jpg )

水平方向にランダムに反転: horizontal_flip

![result_horizontal_flip](https://user-images.githubusercontent.com/28590220/29112609-60ac52e8-7d29-11e7-9ee5-0721e46e435e.jpg )

垂直方向にランダムに反転: vertical_flip

![result_vertical_flip](https://user-images.githubusercontent.com/28590220/29112615-60d31aae-7d29-11e7-90bb-cc5a77c200d8.jpg )

平均の正規化: samplewise_center

![result_samplewise_center](https://user-images.githubusercontent.com/28590220/29112610-60adb78c-7d29-11e7-8ff5-6cd46e0c4b1c.jpg)













