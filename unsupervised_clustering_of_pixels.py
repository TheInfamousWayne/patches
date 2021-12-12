import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import ipdb
from gmm_mml import GmmMml
plt.interactive(True)
path = Path(os.getenv("PATH_TRAIN"))


def load_all_images(images_folder_path):
    image_list = []
    for class_folder in images_folder_path.glob("*"):
        if not os.path.isdir(class_folder):
            continue
        for img_path in class_folder.glob("*.png"):
            img = np.asarray(Image.open(img_path))  # 20x20
            image_list.append(img)
    return image_list


def load_images_per_class(images_folder_path):
    images_per_class = {}
    for class_folder in images_folder_path.glob("*"):
        if not os.path.isdir(class_folder):
            continue
        images_per_class[class_folder] = []
        for img_path in class_folder.glob("*.png"):
            img = np.asarray(Image.open(img_path))  # 20x20
            images_per_class[class_folder].append(img)
    return images_per_class


def get_pixel_array(image_list):
    pixel_array = [i.reshape(-1) for i in image_list]
    pixel_data = np.concatenate(pixel_array)
    return pixel_data


def plot_hist(image_list, title=""):
    pixel_data = get_pixel_array(image_list)
    plt.hist(pixel_data, bins=250)
    plt.title(title)
    plt.show()


show_plots = False

# hist of all image pixels
images = load_all_images(path / "images")
if show_plots:
    plot_hist(images)

# hist of class-wise images
class_wise_images = load_images_per_class(path / "images")
for class_label, images in class_wise_images.items():
    if show_plots:
        plot_hist(images, class_label)

# Fitting Finite Gaussian Mixture Models on (x,) --> pixel_array
X = get_pixel_array(images)
clf = GmmMml(plots=True, live_2d_plot=True)
clf.fit(X.reshape(-1, 1))

#%%
# Fitting GmmMml on (x,y) --> pixel_array, class
X = []
y = []
for class_label, images in class_wise_images.items():
    pixels = get_pixel_array(images).reshape(-1, 1)
    X.append(pixels)
    y.append(np.ones((pixels.shape[0])) * int(class_label.name))

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

idx_upto = 1000

clf2 = GmmMml(plots=True, live_2d_plot=True)
clf2.fit(X[:idx_upto], y[:idx_upto])
