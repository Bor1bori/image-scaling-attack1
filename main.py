# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import load_images
import attack
import time

if __name__ == '__main__':
    source_img, target_img = load_images.load_color_image_from_disk(
        './images/down_cat.jpg',
        './images/down_down_man.jpg')
    start = time.time()
    attack.strong_attack(lambda x: x, source_img, target_img)
    print("time :", time.time() - start)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
