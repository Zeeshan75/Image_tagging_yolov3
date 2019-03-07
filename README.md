========== Image Tagging ===========

Objective :
	 The main objective of the system is to predict the different types of classes/categories 
	 from the given input image.

Process :
	 1. YOLO (You Only Look Once):
	 	Here we have used this model with the latest version of Yolov3 which is predicting the 
	 	80 different classes which are trained on the lots of images. This model is trained on 
	 	the ImageNet dataset. The main advantage of selecting this pre_trained model is as it 
	 	predicts the different classes with in 2-5 sec. 
    
    Note: The weights file is not present in this repo. If you need the weight file you can
    message me in issue with your mail. I will share you the weights file.

Applications:
	 1. It can further be used in generating the captions/sentences for the given input image.
	 2. image recognition is used to translate visual content for blind users and to identify 
	 	inappropriate or offensive images.
	 3. Can be used the predicted tags in Searching the related images from the browser.
	 4. Classification of Images for websites with Large visual Databases.

Run the file as: \n
************************\n
python3 image_tagging.py


output:\n
********************\n
[INFO] loading YOLO from disk...\n
tvmonitor, keyboard\n
person, chair, umbrella\n
person, chair\n
apple, person, diningtable, pottedplant, chair, vase\n
suitcase, handbag, person\n
laptop, keyboard, tvmonitor, person, cup, chair\n
pottedplant, scissors\n
mouse, keyboard, tvmonitor, pottedplant, chair\n
mouse, keyboard\n


