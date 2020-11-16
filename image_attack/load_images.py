import cv2 as cv


def downsize_image(img_src_path: str, size, img_dst_path: str):
    src_image = cv.imread(img_src_path)
    src_image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)

    src_image = cv.resize(src_image, size)
    save_color_image_to_dist(src_image, img_dst_path)

    return


def load_color_image_from_disk(img_src_path: str):
    """
    Loads src, and target RGB images by providing the path to both images.
    """
    # do not forget to swap axis for images loaded by CV, as it saves images in BGR format (not RGB).
    src_image = cv.imread(img_src_path)
    src_image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)

    return src_image


def save_color_image_to_dist(image, save_path: str):
    cv.imwrite(save_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))


if __name__ == '__main__':
    downsize_image('../images/result.jpg', (60, 80), './images/hello.jpg')
