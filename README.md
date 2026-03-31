# img-classifier

## Introduction 

This image classifier is made in order to recognize the 100 classes of a dataset provided in the context of the class Selected Topics in Deep Learning using Visual Recognition, taught at NYCU Hsinchu. 
It is using ResNet50 as a backbone model, and Adam as an optimizer. 

## Environment setup 

This project was made using Spyder 6.0.5, in the Anaconda environment. 

## Usage

To train you model, you have to execute the following line in the console 

``
C:/s8/cv/img-classifier/train.py --wdir
`` 

If your model already exists, you will be asked if you want to 
- Train this process again (Input Y if yes, anything else otherwise)
- Train the 3rd layer aswell (Input Y if yes, anything else otherwise)
- Train the 2nd layer aswell (Input Y if yes, anything else otherwise)
- How many epochs need to be executed (Input a valid integer)
