#!/bin/bash

mkdir -p dataset && \
#wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz && \
tar -xvzf 102flowers.tgz -C dataset
wget -P dataset https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat

# Split the images into folders according to their class
python scripts/split_dataset_by_class.py

# Fixes the missing folders in the test dataset
mkdir 'dataset/test/pink primrose'
mkdir 'dataset/test/prince of wales feathers'
mv 'dataset/train/pink primrose/image_06734.jpg' 'dataset/test/pink primrose/image_06734.jpg'
mv 'dataset/train/prince of wales feathers/image_06850.jpg' 'dataset/test/prince of wales feathers/image_06850.jpg'

# Delete the initial decompressed folder
rm -rf dataset/jpg
