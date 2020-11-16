import cv2 as cv
import random
from image_attack import load_images
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial


def prevention_cropping(img):
    result_img = img[:, :, :]
    for channel in range(3):
        for column in range(1, result_img.shape[1]):
            if random.random() > 0.4:
                result_img[:, column, channel] = img[:, column - 1, channel]

        for row in range(1, result_img.shape[0]):
            if random.random() > 0.4:
                result_img[row, :, channel] = img[row - 1, :, channel]

    return result_img


def downsize_twice(img_src_path: str, size, img_dst_path: str):
    src_image = cv.imread(img_src_path)
    src_image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)

    intermediate_x = int(src_image.shape[1] * ((random.random() / 5) + 0.9))
    intermediate_y = int(src_image.shape[0] * ((random.random() / 5) + 0.9))

    src_image = cv.resize(src_image, (intermediate_x, intermediate_y))
    print(src_image.shape)
    src_image = cv.resize(src_image, size)
    print(src_image.shape)
    load_images.save_color_image_to_dist(src_image, img_dst_path)

    return


def plot_histogram(image1, image1_name, image2, image2_name):
    image1_np = np.array(image1)
    image1_np = np.ravel(image1_np)
    image1_y = [(image1_np == num).sum() / len(image1_np) for num in range(256)]

    image2_np = np.array(image2)
    image2_np = np.ravel(image2_np)
    image2_y = [(image2_np == num).sum() / len(image2_np) for num in range(256)]

    plt.xlabel('Pixel Bins')
    plt.ylabel('Rate of Pixels')
    plt.plot([num for num in range(256)], image1_y)
    plt.plot([num for num in range(256)], image2_y)
    plt.legend([image1_name, image2_name])
    result = 1 - spatial.distance.cosine(image1_y, image2_y)
    print('cos similarity: ', result)
    plt.show()


def plot_scatter(image1, image1_name, image2, image2_name):
    image1_np = np.array(image1)
    image1_np = np.ravel(image1_np)
    image1_y = [0] * 256
    image1_middle_x, image1_middle_y = image1.shape[0] / 2, image1.shape[1] / 2

    for channel in range(3):
        for x in range(image1.shape[0]):
            for y in range(image1.shape[1]):
                distance = spatial.distance.euclidean((image1_middle_x, image1_middle_y), (x, y))
                image1_y[image1[x, y, channel]] += distance
    image1_y = [image1_y[num] / (image1_np == num).sum() for num in range(256)]

    image2_np = np.array(image2)
    image2_np = np.ravel(image2_np)
    image2_y = [0] * 256
    image2_middle_x, image2_middle_y = image2.shape[0] / 2, image2.shape[1] / 2

    for channel in range(3):
        for x in range(image2.shape[0]):
            for y in range(image2.shape[1]):
                distance = spatial.distance.euclidean((image2_middle_x, image2_middle_y), (x, y))
                image2_y[image2[x, y, channel]] += distance
    image2_y = [image2_y[num] / (image2_np == num).sum() for num in range(256)]

    plt.xlabel('Pixel Bins')
    plt.ylabel('Rate of Pixels')
    plt.plot([num for num in range(256)], image1_y)
    plt.plot([num for num in range(256)], image2_y)
    plt.legend([image1_name, image2_name])
    result = 1 - spatial.distance.cosine(image1_y, image2_y)
    print('cos similarity: ', result)
    plt.show()

if __name__ == '__main__':
    image1 = load_images.load_color_image_from_disk('../images/result_scaled.jpg')
    image2 = load_images.load_color_image_from_disk('../images/result_scaled.jpg')
    plot_scatter(image1, 'cat_origin', image2, 'cat_output')
    # plot_histogram(image1, 'cat_origin', image2, 'cat_output')
    # image = load_images.load_color_image_from_disk('./images/result.jpg')
    # image = prevention_cropping(image)
    # load_images.save_color_image_to_dist(image, './images/cropped_result.jpg')
    # load_images.downsize_image('./images/cropped_result.jpg', (100, 80), './images/result_cropped_scaled.jpg')
    # load_images.downsize_image('./images/result.jpg', (100, 80), './images/result_scaled.jpg')
    # load_images.downsize_image('./images/cat2.jpg', (100, 80), './images/cat2_scaled.jpg')
    # downsize_twice('./images/result.jpg', (100, 80), './images/result_downsized_twice.jpg')
