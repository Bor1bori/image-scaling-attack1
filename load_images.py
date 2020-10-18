import cv2 as cv


def load_color_image_from_disk(img_src_path: str, img_tar_path: str):
    """
    Loads src, and target RGB images by providing the path to both images.
    """
    # do not forget to swap axis for images loaded by CV, as it saves images in BGR format (not RGB).
    src_image = cv.imread(img_src_path)
    src_image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)
    tar_image = cv.imread(img_tar_path)
    tar_image = cv.cvtColor(tar_image, cv.COLOR_BGR2RGB)

    return src_image, tar_image
