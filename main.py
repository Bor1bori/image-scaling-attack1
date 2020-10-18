# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import load_images
import attack

if __name__ == '__main__':
    source_img, target_img = load_images.load_color_image_from_disk('./images/cat1.jpg', './images/man1.jpg')
    attack.strong_attack(lambda x: x, source_img, target_img)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
