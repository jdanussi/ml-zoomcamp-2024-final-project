#!/bin/bash

mkdir -p dataset && \
wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz && \
tar -xvzf 102flowers.tgz -C dataset
wget -P dataset https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat

python scripts/split_dataset_by_class.py
rm -rf dataset/jpg
