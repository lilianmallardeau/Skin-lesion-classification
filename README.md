# Skin lesion classification
Skin lesion binary classification using Keras and the [ISIC 2020 dataset](https://challenge2020.isic-archive.com).


## Setup the Python environment and download the dataset
All the required packages can be installed with pip:
```
pip install -r requirements.txt
```
It's better to use a virtual env to prevent version conflicts between packages.

Then you'll have to download the ISIC 2020 train dataset as well as the metadata as CSV files. This can be done automatically with the `setup_dataset.sh` script:
```
./setup_dataset.sh
```


## How to train a model
```
./train.py [--remove-artifacts] [--segmentation] [--checkpoint-folder FOLDER] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
```

Available options:
- `--remove-artifacts` to perform artifacts removal with morphological closing
- `--segmentation` to segment the images with the K-means algorithm
- `--checkpoint-folder` folder to save model checkpoints and final model. Default is `checkpoints/`
- `--epochs` maximum number of epochs to train for. Default is 300
- `--batch-size` batch size to use for training. Default is 256
- `--notifier-prefix` header to send in Telegram messages when sending training progress
- `--help`, `-h` show available options


## How make a prediction using a trained model
Pretrained models can be downloaded in the [Releases](https://github.com/lilianmallardeau/Skin-lesion-classification/releases) page.
```
./test.py image model [--segment] [--remove-artifacts]
```
`image` must be the path to the image to evaluate  
`model` must be the path of the saved Keras model

The output is the probability that the input image is malignant.

---

[Project report](https://mallarde.iiens.net/NTNU/Image_processing_and_analysis_report.pdf)