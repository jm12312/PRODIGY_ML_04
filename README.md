# PRODIGY_ML_04
This repository contains code and resources for the Prodigy ML task 4, which involves gesture recognition using deep learning techniques.
In this task, we perform gesture recognition using a Convolutional Neural Network (CNN). The dataset used for this task is the LeapGestRecog dataset from Kaggle, which contains various hand gestures.

## Dataset
The dataset is downloaded from Kaggle LeapGestRecog Dataset. It consists of 10 classes of gestures, each represented by images.


## Downloading the Dataset:

Python

```
!pip install opendatasets --quiet
import opendatasets as od
dataset_url = 'https://www.kaggle.com/datasets/gti-upm/leapgestrecog'
od.download(dataset_url)
```

## Data Preparation:

Images are resized to 128x128 pixels.
Images are converted to grayscale and normalized.
Labels are created based on gesture names.

## Model Definition:
A Convolutional Neural Network (CNN) with 5 convolutional layers and a classifier is defined.

## Model Training:

The model is trained using the training dataset.
Validation is performed to monitor the model's performance.

## Evaluation:
The trained model is evaluated on the validation dataset.
Confusion matrix is plotted to visualize the performance.

## Results
Accuracy = 99.88 %  
The model achieved high accuracy on the validation dataset with minimal loss. The confusion matrix shows the performance of the model across different gesture classes.


Requirements
Python 3.x
Jupyter Notebook
PyTorch
torchvision
matplotlib
seaborn
scikit-learn
PIL (Python Imaging Library)
Usage


## Clone the repository:

```
bash
git clone https://github.com/jm12312/PRODIGY_ML_04
cd PRODIGY_ML_04
```
