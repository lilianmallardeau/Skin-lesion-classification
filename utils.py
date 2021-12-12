import numpy as np
import cv2
import skimage.morphology

import matplotlib.pyplot as plt


def pictures_grid(args: list, layout: tuple, fontsize:int=12, show_plot:bool=False, return_figure:bool=False):
    """
    Draw a picture grid with layout = (layout[0], layout[1])
    """
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(8, 8))
    ax = axes.flatten()
    [a.set_axis_off() for a in ax]
    for i, img in enumerate(args):
        if type(img) in (list, tuple):
            kwargs = img[2] if len(img) >= 3 else {}
            ax[i].imshow(img[0], **kwargs)
            ax[i].set_title(img[1], fontsize=fontsize)
        else:
            ax[i].imshow(img)
    fig.tight_layout()
    if show_plot:
        fig.show()
    if return_figure:
        return fig


# Images loading
def load_image(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Morphological closing
def apply_morpho_closing(image, disk_size=4):
    disk = skimage.morphology.disk(disk_size)
    r = skimage.morphology.closing(image[..., 0], disk)
    g = skimage.morphology.closing(image[..., 1], disk)
    b = skimage.morphology.closing(image[..., 2], disk)
    return np.stack((r, g, b), axis=-1)


# KMeans segmentation
def kmeans_mask(image, return_rgb=False):
    K = 2
    attempts = 1
    _, labels, centers = cv2.kmeans(np.float32(image.reshape((-1, 3))), K, None, None, attempts, cv2.KMEANS_RANDOM_CENTERS) # or cv2.KMEANS_PP_CENTERS
    centers = np.uint8(centers)
    lesion_cluster = np.argmin(np.mean(centers, axis=1))
    lesion_mask = labels.flatten() == lesion_cluster
    if return_rgb:
        rgb_mask = np.zeros(image.shape)
        rgb_mask[~lesion_mask.reshape(image.shape[:2])] = 255
        return rgb_mask
    return lesion_mask

def kmeans_segmentation(image, force_copy=True, mask=None):
    lesion_mask = mask if mask else kmeans_mask(image)
    segmented_img = image.reshape((-1, 3))
    if force_copy and segmented_img.base is image:
        segmented_img = segmented_img.copy()
    segmented_img[~lesion_mask] = 255
    return segmented_img.reshape(image.shape)


# Data augmentation
def augment_image(image):
    augmented_images = []
    vertical_flip = cv2.flip(image, 0)
    horizontal_flip = cv2.flip(image, 1)
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
    augmented_images.append(vertical_flip)
    augmented_images.append(horizontal_flip)
    augmented_images.append(cv2.rotate(vertical_flip, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(horizontal_flip, cv2.ROTATE_90_CLOCKWISE))
    return augmented_images
