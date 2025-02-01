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
In this project, I trained a Convolutional Neural Network (CNN) model for flower classification, leveraging transfer learning with the Xception architecture pretrained on ImageNet. The trained model was then deployed as a public cloud service on AWS, utilizing Lambda for serverless inference and API Gateway for external access. This setup allows users to send image URLs and receive predictions without the need for dedicated infrastructure. The solution is both scalable and cost-effective, accessible via a simple API request.

The dataset used is the Oxford 102 Flower dataset that can be download from [here](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz)


## 2. Dependency and environment management
**Pipenv** was used to create the virtual environment and install the dependencies. In order to follow the development of the project you must clone [this repository](https://github.com/jdanussi/ml-zoomcamp-2024-final-project.git), create the virtual environment installing the required dependencies and activate it as demonstrated below.

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


## 3. Exploratory Data Analysis
The dataset used consists of 102 flower categories. Each class consists of between 40 and 258 images. 
The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.

In the [Exploratory Data Analysis (EDA)](notebook.ipynb#exploratory-data-analysis-eda) section of the notebook, it is observed that the classes are not balanced. 
Since we are using transfer learning with a pre-trained model, which has already learned features from a large, balanced dataset (ImageNet), we can expect that the fine-tuning performed on top of the convolutional layers will lead to better generalization, particularly for minority classes.

The [dataset](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz) and the [imagelabels](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat) were downloaded from the [Oxford 102 Flower dataset Homepage](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/)

The names for classes where obteined from [JosephKJ/Oxford-102_Flower_dataset_labels.txt](https://gist.github.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1) and used into the python script `split_dataset_by_class.py` that was used as a helper to split the dateset into train, validation and test. 


## 4. Data preparation
You don't need to run the procedure detailed below, as the dataset with images split by class is already included in this repository. This is just for informational purposes.

```bash

# Download the dataset and the image labels, create the dataset folder structure and split the images into classes
> bash dataset-setup.sh

```

The following part of the bash script:

```python

mkdir 'dataset/test/pink primrose'
mkdir 'dataset/test/prince of wales feathers'
mv 'dataset/train/pink primrose/image_06734.jpg' 'dataset/test/pink primrose/image_06734.jpg'
mv 'dataset/train/prince of wales feathers/image_06850.jpg' 'dataset/test/prince of wales feathers/image_06850.jpg'

```

was added because, after executing the Python script `split_dataset_by_class.py`, it was found that two class folders were missing in the dataset/test directory. Therefore, the missing folders were created and populated with at least one file from the training set. This fix was necessary to avoid dimensional problems when using the test dataset during the evaluation of the model.

The split of the dataset was also published in the [flowers-dataset](git@github.com:jdanussi/flowers-dataset.git) repository, which will be useful for testing the services we deploy.



## 5. Model training and tuning
We will use the **Xception** model from **Keras**, which was pre-trained on **ImageNet**, to extract image features. Then, we will build and train a dense model on top of it using transfer learning.

The model was tuned on Saturn Cloud Platform (https://saturncloud.io/), leveraging its **GPU-supported Jupyter notebooks** to accelerate experimentation. The tuning process involved testing different values for the *learning rate*, *dropout rate*, and *inner layer size*, as well as evaluating the impact of *data augmentation*. The best performance was achieved with the following configuration:

- Learning rate: 0.001  
- Dropout rate: 0.0 (no dropout)  
- Inner layer size: 1000  
- No data augmentation

The most performant model was first saved in TensorFlow format (h5) and then converted to TensorFlow Lite (tflite) format to be used in a Lambda function without the need to import the heavy TensorFlow library. The best model was saved and converted to the file `xception_v1_1_18_0.924.tflite`.

The process and results can be seen in the [Model training and tuning](notebook.ipynb#model-training-and-tuning) section of the notebook.


## 6. Lambda function
Following the approach from the Zoomcamp, the prediction service was deployed using AWS Lambda, a serverless service. The goal is to create an endpoint that, upon receiving a URL of a flower image, returns the top 10 most probable classes the image belongs to, along with their respective scores.

To completely remove the dependency on `TensorFlow`, `TensorFlow Lite Runtime` was installed, along with `keras-image-helper`, which replaces the `load_img` and `preprocess_input` functions from Keras.

We test the Lambda function locally, using ipython 

```python

In [1]: import lambda_function
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

In [2]: event = {'url': 'https://github.com/jdanussi/flowers-dataset/blob/main/test/bee%20balm/image_03060.jpg?raw=true'}

In [3]: result = lambda_function.lambda_handler(event, None)

# Show the top 10 most probable classes
In [4]: print(dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]))
{'bee balm': 14.12916088104248, 'common dandelion': 3.8607218265533447, 'globe thistle': -0.3712370991706848, 'cape flower': -1.3991206884384155, 'red ginger': -1.9034096002578735, 'carnation': -2.3492584228515625, 'blanket flower': -3.5331532955169678, 'azalea': -4.272549629211426, 'gaura': -4.346775531768799, 'sweet william': -4.356930732727051}


In [5]: event = {'url': 'https://github.com/jdanussi/flowers-dataset/blob/main/test/bolero%20deep%20blue/image_07132.jpg?raw=true'}

In [6]: result = lambda_function.lambda_handler(event, None)

# Show the top 10 most probable classes
In [7]: print(dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]))
{'bolero deep blue': -0.8681230545043945, 'canterbury bells': -1.2452996969223022, 'spring crocus': -1.2499873638153076, 'corn poppy': -2.2304394245147705, 'bearded iris': -2.652745485305786, 'sweet pea': -3.239044666290283, 'cyclamen': -3.615382194519043, 'tree mallow': -3.7148847579956055, 'lotus': -5.3969621658325195, 'sword lily': -5.507812976837158}

```


## 7. Local model deployment with Docker
Before deploying the Lambda function on AWS, we package everything into a Docker container using the AWS base image `public.ecr.aws/lambda/python:3.9`.
The `tflite-runtime` package was replaced with a version compiled by Alexey (github.com/alexeygrigorev/tflite-aws-lambda) for the correct Linux version used in the AWS environment.

The following demonstrates how to build the image and deploy the container with the service provided by the Lambda function. Then, we test the service using the script `test_locally.py`, which passes an image URL from the [flower-dataset](git@github.com:jdanussi/flowers-dataset.git).

```bash

# Build the docker image
> docker build -t flowers-classification .

# Check the image created
> docker image ls | grep flowers-classification
flowers-classification    latest    5c0395b867f0   7 minutes ago   839MB

# Deploy a lambda_function using a docker container
> docker run -it --rm -p 8080:8080 flowers-classification:latest
01 Feb 2025 15:23:47,084 [INFO] (rapid) exec '/var/runtime/bootstrap' (cwd=/var/task, handler=)

```

and test the service

```bash

# Test from other terminal of the same instance
> python test_locally.py 
{'cape flower': 19.202898025512695, 'gaura': 3.7901079654693604, 'trumpet creeper': 1.7330631017684937, 'blackberry lily': -0.5172544717788696, 'columbine': -1.182721734046936, 'tiger lily': -2.336095094680786, 'cautleya spicata': -2.3456387519836426, 'orange dahlia': -3.425830364227295, 'pink quill': -3.491724967956543, 'fire lily': -3.7202038764953613}
>

# Docker outputs the request
> docker run -it --rm -p 8080:8080 flowers-classification:latest
01 Feb 2025 15:23:47,084 [INFO] (rapid) exec '/var/runtime/bootstrap' (cwd=/var/task, handler=)
START RequestId: fd691123-d69f-4998-8ab5-19c219cd0caa Version: $LATEST
01 Feb 2025 15:23:52,576 [INFO] (rapid) INIT START(type: on-demand, phase: init)
01 Feb 2025 15:23:52,576 [INFO] (rapid) The extension's directory "/opt/extensions" does not exist, assuming no extensions to be loaded.
01 Feb 2025 15:23:52,576 [INFO] (rapid) Starting runtime without AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN , Expected?: false
01 Feb 2025 15:23:52,803 [INFO] (rapid) INIT RTDONE(status: success)
01 Feb 2025 15:23:52,803 [INFO] (rapid) INIT REPORT(durationMs: 227.116000)
01 Feb 2025 15:23:52,803 [INFO] (rapid) INVOKE START(requestId: 0673d8a5-a7bf-4b08-9d9b-f4156281b196)
01 Feb 2025 15:23:53,534 [INFO] (rapid) INVOKE RTDONE(status: success, produced bytes: 0, duration: 730.686000ms)
END RequestId: 0673d8a5-a7bf-4b08-9d9b-f4156281b196
REPORT RequestId: 0673d8a5-a7bf-4b08-9d9b-f4156281b196  Init Duration: 0.05 ms  Duration: 957.99 ms     Billed Duration: 958 ms Memory Size: 3008 MB    Max Memory Used: 3008 MB

```


## 8. Cloud service deployment with AWS Lambda and Amazon API Gateway
The **Flower Class Prediction** service was deployed on **AWS Lambda**, a serverless computing platform, using a containerized approach. The Lambda function was created from a previously built Docker image. Before provisioning AWS Lambda, the Docker image was registered to a private repository in **Amazon Elastic Container Registry (Amazon ECR)**, allowing seamless selection during the function setup.

Additionally, the default configuration of the **Lambda** function was modified to allocate **1024 MB of memory** and set a **timeout of 30 seconds** to optimize execution performance.


Finally, the service provided by Lambda was exposed through a RESTful API endpoint via Amazon API Gateway, which only exposes the `predict` resource, accessible through the POST method.

The API service is public and available via the following endpoint:
https://exy970w5sd.execute-api.us-east-1.amazonaws.com/test/predict


Below is the test of the service, which was performed using the test_api_gateway.py script.

```bash

# Make sure your AWS credentials are properly set for CLI access. 
# I do this by exporting the `AWS_PROFILE` environment variable, setting it to use the appropriate profile from `~/aws/credentials`.
# Replace "aws-profile" with the appropriate value for your case.
> export AWS_PROFILE=my-aws-profile

# You need this to be able to login to this registry with docker
> $(aws ecr get-login --no-include-email --region us-east-1)
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
Login Succeeded

# Retag the docker image to prepare for ECR
# Replace <YOUR-AWS-ACCOUNT-ID> with you AWS ACCOUNT ID
ACCOUNT=<YOUR-AWS-ACCOUNT-ID>
REGION=us-east-1
REGISTRY=flowers-tflite-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
TAG=flowers-model-v1-002
REMOTE_URI=${PREFIX}:${TAG} 

> docker tag flowers-classification:latest ${REMOTE_URI}

# Push image to ECR (The example is not the firsf push, that's why there layers that already exists)
> docker push ${REMOTE_URI}
The push refers to repository [564443450717.dkr.ecr.us-east-1.amazonaws.com/flowers-tflite-images]
2de9bed70fa9: Pushed 
761d3ed3c3f5: Layer already exists 
ba79f31df61d: Layer already exists 
9ab1906c8c2c: Layer already exists 
1254b1310dfa: Layer already exists 
ea2aba827ecd: Layer already exists 
0091b0f618e4: Layer already exists 
50467525f7a2: Layer already exists 
a509bd9469d3: Layer already exists 
3fda89dc2c05: Layer already exists 
flowers-model-v1-002: digest: sha256:1c54bff1e4e7553321be248f2a94aaf441353b2c66985624a4dd8a9acb849594 size: 2425

```

After pushing the Docker image, I created the Lambda function and API Gateway endpoint from the AWS console. 

With the service endpoint available on AWS, testing can be performed from the local terminal using the script `test_api_gateway.py`.

```bash

# Testing from terminal using script test_api_gateway.py
> python test_api_gateway.py 
{'cape flower': 19.202898025512695, 'gaura': 3.7901079654693604, 'trumpet creeper': 1.7330631017684937, 'blackberry lily': -0.5172544717788696, 'columbine': -1.182721734046936, 'tiger lily': -2.336095094680786, 'cautleya spicata': -2.3456387519836426, 'orange dahlia': -3.425830364227295, 'pink quill': -3.491724967956543, 'fire lily': -3.7202038764953613}
> 

```

Finally, some screenshots are provided, showing the most relevant settings for each service.

Saturn Cloud environment config
![saturncloud-env](images/saturncloud-env.png)

Elastic Container Registry **flowers-tflite-images**
![ecr](images/ecr.png)

Selecting a Docker image during Lambda function creation
![lambda-image](images/lambda-image.png)

Lambda configuration
![lambda-config](images/lambda-config.png)

Lambda testing: **Test event**
![lambda-test-01](images/lambda-test-01.png)

Lambda testing: **Result Details**
![lambda-test-02](images/lambda-test-02.png)

API Gateway **Method request**
![apigw-01](images/apigw-01.png)

API Gateway **Integration request**
![apigw-02](images/apigw-02.png)