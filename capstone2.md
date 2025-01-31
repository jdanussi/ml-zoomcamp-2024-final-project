

(base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ pipenv install
Creating a virtualenv for this project...
Pipfile: /home/jdanussi/Documents/DataTalksClub/ml-zoomcamp/capstone2/Pipfile
Using default python from /usr/bin/python3 (3.10.12) to create virtualenv...
⠋ Creating virtual environment...created virtual environment CPython3.10.12.final.0-64 in 614ms
  creator CPython3Posix(dest=/home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/jdanussi/.local/share/virtualenv)
    added seed packages: pip==24.3.1, setuptools==75.6.0, wheel==0.45.1
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

✔ Successfully created virtual environment!
Virtualenv location: /home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX
Creating a Pipfile for this project...
Pipfile.lock not found, creating...
Locking [packages] dependencies...
Locking [dev-packages] dependencies...
Updated Pipfile.lock (fedbd2ab7afd84cf16f128af0619749267b62277b4cb6989ef16d4bef6e4eef2)!
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
Installing dependencies from Pipfile.lock (e4eef2)...
(base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ pipenv shell
Launching subshell in virtual environment...
 . /home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX/bin/activate
(base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$  . /home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX/bin/activate
(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ python -V
Python 3.10.12
(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ which python
/home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX/bin/python
(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ 

(base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ pipenv install
Creating a virtualenv for this project...
Pipfile: /home/jdanussi/Documents/DataTalksClub/ml-zoomcamp/capstone2/Pipfile
Using default python from /usr/bin/python3 (3.10.12) to create virtualenv...
⠋ Creating virtual environment...created virtual environment CPython3.10.12.final.0-64 in 614ms
  creator CPython3Posix(dest=/home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/jdanussi/.local/share/virtualenv)
    added seed packages: pip==24.3.1, setuptools==75.6.0, wheel==0.45.1
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

✔ Successfully created virtual environment!
Virtualenv location: /home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX
Creating a Pipfile for this project...
Pipfile.lock not found, creating...
Locking [packages] dependencies...
Locking [dev-packages] dependencies...
Updated Pipfile.lock (fedbd2ab7afd84cf16f128af0619749267b62277b4cb6989ef16d4bef6e4eef2)!
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
Installing dependencies from Pipfile.lock (e4eef2)...
(base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ pipenv shell
Launching subshell in virtual environment...
 . /home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX/bin/activate
(base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$  . /home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX/bin/activate
(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ python -V
Python 3.10.12
(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ which python
/home/jdanussi/.local/share/virtualenvs/capstone2-Acj_LonX/bin/python
(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ 

pipenv install jupyter --dev
jupyter notebook

Create a new notebook notebook/notebook.ipynb
Install dependencies from the notebook
!pipenv install tensorflow

Install the Oxford flower dataset from the shell environment
pipenv install tensorflow_datasets # or from inside the notebook !pipenv install tensorflow_datasets


!pipenv install scipy

!wegt https://github.com/hadeelbkh/classifier-files/blob/master/label_map.json

pipenv install matplotlib


# Dataset

```bash
mkdir flowers-dataset
cd flowers-dataset
wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz
tar -xvf 102flowers.tgz

wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat

touch python split_dataset_by_class.py
```

The python scipt to split the dataset in folders

```python
import os
import shutil
import random
from scipy.io import loadmat

# Paths
source_dir = "jpg"  # Folder containing the images
output_dir = "splits"  # Destination folder for splits
labels_path = "imagelabels.mat"  # Path to imagelabels.mat

# Create base output directories
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "validation")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load imagelabels.mat
labels = loadmat(labels_path)['labels'].flatten()  # Class labels (1-indexed)

class_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", 
    "wild geranium", "tiger lily", "moon orchid", "bird of paradise", "monkshood", 
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", 
    "yellow iris", "globe flower", "purple coneflower", "peruvian lily", 
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", 
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", 
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", 
    "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", 
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", 
    "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", 
    "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", 
    "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", 
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", 
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy", 
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", 
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", 
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", 
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", 
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", 
    "bee balm", "pink quill", "foxglove", "bougainvillea", "camellia", 
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", 
    "blackberry lily"
]

# Get all image indices
all_indices = list(range(1, len(labels) + 1))  # MATLAB indices are 1-based

# Shuffle and split indices into 80/10/10 for train/validation/test
random.seed(42)  # Ensure reproducibility
random.shuffle(all_indices)

train_split = int(0.8 * len(all_indices))
val_split = int(0.9 * len(all_indices))

train_indices = all_indices[:train_split]
val_indices = all_indices[train_split:val_split]
test_indices = all_indices[val_split:]

# Helper function to copy images to their respective class folders
def copy_images_by_class(indices, destination):
    for idx in indices:
        class_label = labels[idx - 1]  # Class label for the image (1-indexed in MATLAB)
        #class_dir = os.path.join(destination, f"class_{class_label:03d}")
        class_dir = os.path.join(destination, class_names[class_label - 1])

        os.makedirs(class_dir, exist_ok=True)  # Create class directory if it doesn't exist

        filename = f"image_{idx:05d}.jpg"  # Format index as image_XXXX.jpg
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(class_dir, filename)
        if os.path.exists(src_path):  # Check if file exists
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")

# Organize images into train, validation, and test folders by class
copy_images_by_class(train_indices, train_dir)
copy_images_by_class(val_indices, val_dir)
copy_images_by_class(test_indices, test_dir)

print("Dataset split and organized by class completed!")
```

Run the script to split the images in folders
```bash
python split_dataset_by_class.py
```

Delete jpg folder
```bash
rm -rf jpg
```

# To use the same split used in https://www.tensorflow.org/datasets/catalog/oxford_flowers102 follow this steps
wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/setid.mat
python split_dataset_setid.py # see the script detais from file
python split_dataset_setid.py


(base) jdanussi@jad-xps15:~/Dropbox/openvpn/bcp/aws-cvpn-endpoint$ lspci | grep -i nvidia
01:00.0 3D controller: NVIDIA Corporation GP107M [GeForce GTX 1050 Mobile] (rev a1)

CUDA Toolkit Support 8.0–11.3


git init 
git add .
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:jdanussi/capstone2.git
git push -u origin main


# CUDA
Cuda 12 + tf-nightly 2.12: Could not find cuda drivers on your machine, GPU will not be used, while every checking is fine and in torch it works
https://stackoverflow.com/questions/75614728/cuda-12-tf-nightly-2-12-could-not-find-cuda-drivers-on-your-machine-gpu-will

How to find the right Cuda version?
https://knowmledge.com/2023/11/18/ml-zoomcamp-2023-deep-learning-part-2/

How to install CUDA, cuDNN and TensorFlow on Ubuntu 22.04 (2023)
https://medium.com/@gokul.a.krishnan/how-to-install-cuda-cudnn-and-tensorflow-on-ubuntu-22-04-2023-20fdfdb96907

En resumen

sudo apt update
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt update

sudo apt install cuda-11-8
sudo reboot

sudo apt autoremove

register in NVIDEA, find the tar to doanload, extract, rename the result folder to cudnn and
cd ~/Downloads/cudnn
sudo cp -P include/cudnn* /usr/local/cuda-11.8/include
sudo cp -P lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

sudo vi /etc/.bashrc

export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


export AWS_PROFILE=jad-personal

aws ecr create-repository --repository-name flowers-tflite-images --region us-east-1
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:564443450717:repository/flowers-tflite-images",
        "registryId": "564443450717",
        "repositoryName": "flowers-tflite-images",
        "repositoryUri": "564443450717.dkr.ecr.us-east-1.amazonaws.com/flowers-tflite-images",
        "createdAt": 1738279265.518,
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}


(capstone2) (base) jdanussi@jad-xps15:~/Documents/DataTalksClub/ml-zoomcamp/capstone2$ $(aws ecr get-login --no-include-email --region us-east-1)
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
Login Succeeded

ACCOUNT=564443450717
REGION=us-east-1
REGISTRY=flowers-tflite-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

TAG=flowers-model-v1-001
REMOTE_URI=${PREFIX}:${TAG}


# Test lambda
{
    "url": "https://raw.githubusercontent.com/jdanussi/flowers-dataset/refs/heads/main/test/cape%20flower/image_03810.jpg"
}

# Test API Gateway
“Request Body”. Type
{"url": "https://raw.githubusercontent.com/jdanussi/flowers-dataset/refs/heads/main/test/cape%20flower/image_03810.jpg"}

https://exy970w5sd.execute-api.us-east-1.amazonaws.com/test




# dataset init
bash dataset-setup.sh
ls -1 dataset/train > dat_train.txt
ls -1 dataset/validation > dat_val.txt
ls -1 dataset/test > dat_test.txt
diff dat_train.txt dat_val.txt
diff dat_train.txt dat_test.txt