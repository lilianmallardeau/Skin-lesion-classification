#!/usr/bin/env bash
set -e

[ ! -d "dataset/" ] || mkdir dataset
cd dataset

echo "Downloading ISIC 2020 dataset..."
wget "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
wget "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"
wget "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
unzip ISIC_2020_Training_JPEG.zip
mv train train_jpeg
