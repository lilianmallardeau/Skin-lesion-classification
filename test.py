#!/usr/bin/env python3
import argparse

import numpy as np
import cv2

import keras

from utils import *

input_shape = (224, 224, 3)


parser = argparse.ArgumentParser()
parser.add_argument("image", help="Image to evaluate", type=str)
parser.add_argument("model", help="Model to use for classification", type=str)
parser.add_argument("--segment", action="store_true", help="Segment the image before classification")
parser.add_argument("--remove-artifacts", action="store_true", help="Apply morphological closing before classification")
args = parser.parse_args()

print("Loading model...")
model = keras.models.load_model(args.model)

print("Loading image...")
image = cv2.resize(load_image(args.image), input_shape[:2])

if args.remove_artifacts:
    image = apply_morpho_closing(image, disk_size=1)

if args.segment:
    image = kmeans_segmentation(image, force_copy=False)

prediction = model.predict(image[np.newaxis]).ravel()[0]

print(f"{{}} (malignant probability: {prediction:%})".format(
    'Benign' if prediction < 0.5 else 'Malignant'
))
