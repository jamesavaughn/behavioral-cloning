## **Behavioral Cloning P3 **

Writeup - James Vaughn - Behavioral Cloning - P3



#The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior - ***[DONE - See driving_log.csv file]***
* Build, a convolution neural network in Keras that predicts steering angles from images ***- [DONE - See file named 'model2.py']***
* Train and validate the model with a training and validation set ***- [DONE - See file named 'model.h5']***
* Test that the model successfully drives around track one without leaving the road ***- [DONE - See video file named 'run1.mp4']***
* Summarize the results with a written report ***- [DONE - See file named Readme_JamesVaughn.md]***

[//]: # (Image References)

[image1]: ./examples/placeholder_model.pgn "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model2.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network
* **Readme_JamesVaughn.md** summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The **model2.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses Nvidia's Self-Driving Car Model Architecture.

* Layer 1: Convolutional Layer | 5x5 filter size, stride 2, relu nonlinearity activation
* Layer 2: Convolutional Layer | 5x5 filter size, stride 2, relu activation
* Layer 3: Convolutional Layer | 5x5 filter size, stride 2, relu activation
* Layer 4: Convolutional Layer | 3x3 filter size, relu activation
* Layer 5: Convolutional Layer | 3x3 filter size, relu activation
* Layer 6: Flatten Layer
* Layer 6: Fully Connected Layer size 100
* Layer 7: Fully Connected Layer size 50
* Layer 8: Fully Connected Layer size 10
* Layer 9: Fully Connected Layer size 1

See (model2.py lines 60-73)


Data were preprocessed using the following steps:

1. The data were normalized in the model using a Keras lambda layer (code line 62).
2. The data were augmented by flipping the images to better balance the data (left and right)
3. Top of the images was copped to remove background and focus on the road lanes
4. Bottom of the was cropped to remove hood of the car and focus on the road lanes
5. Removed all 'zero' measurements from the dataset to drive better straight
6. Added a correction factor to the measurements to balance over/under steering 


####2. Attempts to reduce overfitting in the model

I did not use dropout layers.  The model was trained and validated on different datasets to avoid overfitting (code line 76) and data was shuffled.  Model was tested by running the simulator in autonomous mode and ensuring that the vehicle could stay on the track.


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model2.py line 75).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drove the course 4 times, staying centered.  I drove a recovery course alternating recovering from the left and the right sides of the road.  


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a basic convolution neural network LeNet model.   I thought this model might be appropriate as a starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. After running the model I found that the vehicle could not stay on the road.

Then I researched the Nvida Cov Model and modified the LeNet architecture to match.

I then adjusted the number of epochs and correction factor to fine tune the results.  After each training session, I ran the simulator in autonomous mode.


####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...
* Layer 1: Convolutional Layer | 5x5 filter size, stride 2, relu nonlinearity activation
* Layer 2: Convolutional Layer | 5x5 filter size, stride 2, relu activation
* Layer 3: Convolutional Layer | 5x5 filter size, stride 2, relu activation
* Layer 4: Convolutional Layer | 3x3 filter size, relu activation
* Layer 5: Convolutional Layer | 3x3 filter size, relu activation
* Layer 6: Flatten Layer
* Layer 6: Fully Connected Layer size 100
* Layer 7: Fully Connected Layer size 50
* Layer 8: Fully Connected Layer size 10
* Layer 9: Fully Connected Layer size 1


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

**Sample images can be found in the github repository.**

After collecting data, I had 16,753 images and measurements. 
I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. I ran the model for 10 epochs.
