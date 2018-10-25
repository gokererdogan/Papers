#!/usr/bin/python3

from pathlib import Path
import numpy as np

from skimage.io import imread
from skimage.transform import resize

"""
This script creates numpy arrays from Omniglot dataset that are used to instantiate MxNet data iterators.

Download the background and evaluation zip files from https://github.com/brendenlake/omniglot/
  - https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip
  - https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip
and extract them to data/omniglot folder. (Put alphabets from both dataset into omniglot folder, so all alphabets are
in the same folder.)

Run this script to create numpy arrays of images, alphabet ids, and character ids.
common/get_omniglot function needs the numpy arrays of images to instantiate the MXNet data iterators used for
training/evaluation.
"""

if __name__ == "__main__":
    new_size = (28, 28)
    train_ratio = 0.7
    train_test = np.ones(20)
    train_test[int(20*train_ratio):] = 0

    alphabet_names = []
    train_images = []
    train_image_to_alphabet = []
    train_image_to_character = []
    test_images = []
    test_image_to_alphabet = []
    test_image_to_character = []

    for a_id, alphabet_dir in enumerate(Path('omniglot').iterdir()):
        print(alphabet_dir.name)
        alphabet_names.append(alphabet_dir.name)
        for character_dir in alphabet_dir.iterdir():
            c_id = int(character_dir.name[-2:]) - 1
            print("Character ", c_id)
            np.random.shuffle(train_test)
            for img_file in character_dir.iterdir():
                img_i = int(img_file.stem[-2:]) - 1
                img = (resize(imread(img_file), new_size, anti_aliasing=True, mode='constant') < 0.5).astype(np.float)
                if train_test[img_i]:  # training set
                    train_images.append(img)
                    train_image_to_alphabet.append(a_id)
                    train_image_to_character.append(c_id)
                else:
                    test_images.append(img)
                    test_image_to_alphabet.append(a_id)
                    test_image_to_character.append(c_id)

    train_images = np.stack(train_images, axis=0).reshape(-1, 1, *new_size)
    train_image_to_alphabet = np.stack(train_image_to_alphabet)
    train_image_to_character = np.stack(train_image_to_character)
    test_images = np.stack(test_images, axis=0).reshape(-1, 1, *new_size)
    test_image_to_alphabet = np.stack(test_image_to_alphabet)
    test_image_to_character = np.stack(test_image_to_character)

    np.save('omniglot_train_img.npy', train_images)
    np.save('omniglot_test_img.npy', test_images)

    np.save('omniglot_train_alphabet.npy', train_image_to_alphabet)
    np.save('omniglot_test_alphabet.npy', test_image_to_alphabet)

    np.save('omniglot_train_char.npy', train_image_to_character)
    np.save('omniglot_test_char.npy', test_image_to_character)


