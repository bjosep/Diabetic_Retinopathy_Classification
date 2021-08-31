# Diabetic Retinopathy Classification
![alt text](https://github.com/bjosep/Diabetic_Retinopathy_Classification/blob/main/assets/retina_imgs_.PNG?raw=true)
## About
Diabetic retinopathy is a diabetes complication that affects eyes. It is the leading cause of blindness among working aged adults.

The goal of this project is to build a deep learning model to screen images for disease by classifying them into one of the five possible categories: No Diabetic Retinopathy,  Mild, Moderate, Severe, Profelivative Diabetic Retinopathy


## Demo video
https://youtu.be/U8bEyJfd5dE

## Technologies
* TensorFlow
* OpenCV
* Streamlit

## Model training

I took advantage of transfer learning by adding to a pre-trained DenseNet 121 architecture, 2 layers with respectively 1024 and 512 units and relu activation function then the final layer with 5 units and softmax activation function.

The model was trained for 25 epochs. I used Categorical cross entropy loss function, rmsprop optimizer and a batch size of 16 images. The model training took approximately 2 hours and 25 minutes on one GPU. The reported validation cohen’s Kappa (measures the agreement between the human rater and the model prediction) was 0.78

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
It consists of a large set of retina medical images that were rated by clinicians:
0 for No Diabetic Retinopathy,  1 for Mild, 2 for Moderate, 3 for Severe, and 4 for Profelivative Diabetic Retinopathy

Originally, we were provided 2 files: the first one containing training images, whereas the second one was a table with 2 columns: image_id and label. After splitting the data into training (80%) and validation (20%), I decided to change how it is stored to facilitate the subsequent steps, namely tensorflow data generator. The new structure comprises 2 folders: data/train and data/valid such that each contains 5 subfolders (0,1,2,3,4) where images with the same label are stored.


