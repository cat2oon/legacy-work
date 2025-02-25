import os
import glob
import cv2
import numpy as np


def eye_image_packer(profile_base_path, pack_size=100000, image_shape=(38, 60, 3)):
    block_idx = 0
    imgs_to_pack = []
    for profile_dir in glob.glob("{}/*".format(profile_base_path)):
        for img_path in glob.glob("{}/*.jpg".format(profile_dir)):
            img = cv2.imread(img_path)
            if img is None or img.shape != image_shape:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs_to_pack.append(img)

            if len(imgs_to_pack) != pack_size:
                continue

            npb_path = os.path.join("{}/every-eye-{:03d}.npz".format(profile_base_path, block_idx))
            pack_images(npb_path, imgs_to_pack)
            imgs_to_pack = []
            block_idx += 1


def pack_images(path, images):
    npb = np.array(images)
    np.savez_compressed(path, images=npb)
    print("*** complete write {} ***".format(path))
