# Flowers Classification

# Summary
1. Problem Description
2. Dependency and environment management
3. EDA
4. Data preparation
5. Model training and tuning
6. Lambda function
7. Local service deployment with Docker
8. Cloud service deployment with AWS Lambda and Amazon API Gateway

## 1. Problem Description
In this project, I trained a Convolutional Neural Network (CNN) model for flower classification using transfer learning. The trained model was then deployed as a public cloud service on AWS, utilizing Lambda for serverless inference and API Gateway for external access. This setup enables users to send image URLs and receive predictions without requiring dedicated infrastructure. The solution is scalable, cost-effective, and accessible via a simple API request.

The dataset used is the Oxford 102 Flower dataset that can be download from https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz


## 2. Dependency and environment management
Pipenv was used to create the virtual environment and install the dependencies. In order to follow the development of the project you must clone the [repository](https://github.com/jdanussi/ml-zoomcamp-2024-final-project.git), create the virtual environment installing the required dependencies and activate it as demonstrated below.

```bash

# Clone the project repository
> git clone https://github.com/jdanussi/ml-zoomcamp-2024-final-project.git

# Change dir to the project folder
> cd ml-zoomcamp-2024-final-project

# Create a new virtual environment and install the project dependencies
> pipenv install

# Activate the new environment
> pipenv shell

```


## 3. EDA
The dataset used is a 102 category dataset, consisting of 102 flower categories. Each class consists of between 40 and 258 images. 

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.

In the [Exploratory Data Analysis (EDA)](notebook.ipynb#exploratory-data-analysis-eda) section of the notebook, it is observed that the classes are not balanced. Using transfer learning with a pre-trained model can help, as the model has already learned features on a large, balanced dataset (like ImageNet), which can then be fine-tuned on your imbalanced dataset. This often leads to better generalization, especially for minority classes.

The [dataset](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz) and the [imagelabels](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat) were downloaded from the [Oxford 102 Flower dataset Homepage](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/)

The names for classes where obteined from [JosephKJ/Oxford-102_Flower_dataset_labels.txt](https://gist.github.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1) and used into the python script `scripts/split_dataset_by_class.py` that was used as a helper to split the dateset into train, validation and test. 


## 4. Data preparation
You don't need to run the procedure detailed below, as the dataset with images split by class is already included in this repository. This is just for informational purposes.

```bash

# Download the dataset and the image labels, create the dataset folder structure and split the images into classes
> bash dataset-setup.sh

```

The following part of the bash script:

```bash

mkdir 'dataset/test/pink primrose'
mkdir 'dataset/test/prince of wales feathers'
mv 'dataset/train/pink primrose/image_06734.jpg' 'dataset/test/pink primrose/image_06734.jpg'
mv 'dataset/train/prince of wales feathers/image_06850.jpg' 'dataset/test/prince of wales feathers/image_06850.jpg'

```

was added because, after executing the Python script `scripts/split_dataset_by_class.py`, it was found that two class folders were missing in the dataset/test directory. Therefore, the missing folders were created and populated with at least one file from the training set. This fix was necessary to avoid dimensional problems when using the test dataset during the evaluation of the model.


## 5. Model training and tuning
We will use the Xception model from Keras, which was pre-trained on ImageNet, to extract image features. Then, we will build and train a dense model on top of it using transfer learning.

The model was tuned by testing different values for the learning rate, dropout rate, and inner layer size. Data augmentation was also evaluated. The best performance was achieved with the following parameters:  

- Learning rate: 0.001  
- Dropout rate: 0.0 (no dropout)  
- Inner layer size: 1000  
- No data augmentation

The process and result can be seen in the [Model training and tuning](notebook.ipynb#model-training-and-tuning) section of the notebook.


## 6. Lambda function


## 7. Local model deployment with Docker
Below is the process for exposing the lambda function running inside a Docker container.

```bash

# Build the docker image
> docker build -t flowers-model .

# Deploy a local service for water potability prediction using a docker container
> docker run -it --rm -p bla bla


# Test the containerized service from other terminal of the same instance
> python predict-test.py
{'potability': False, 'potability_probability': 0.20214878729185673}
Water sample id water-230 is Non-potable water
>

```