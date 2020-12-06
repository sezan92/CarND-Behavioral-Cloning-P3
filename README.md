# **Behavioral Cloning** 

## Writeup 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[arch]: ./images/model_arch.png "Model Architecture"
[normal_center]: ./images/center1.jpg "Center Lane Normal"
[normal_recovery1]: ./images/recovery1.jpg "Recovery Image"
[normal_recovery2]: ./images/recovery2.jpg "Recovery Image"
[normal_recovery3]: ./images/recovery3.jpg "Recovery Image"
[difficult1]: ./images/difficult1.jpg "Difficult track Image"
[difficult2]: ./images/difficult2.jpg "Difficult track Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing . The simulator has two tracks.

***For track 1***

```sh
python drive.py model.h5
```

***For track 2***
```sh
python drive.py model_difficult.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. The `Train.ipynb` contains  explainations how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is actually an improved version of `Alexnet`. I used a sequential model of 3 Convolutional Layers with 64,128 and 256 filters each having kernel size of `5x5` . They were followed by three dense layers of `250`,`120`,`84` units . All of the layers use `relu` activation unit. I used Maxpooling layers after each convolutional layer. (lines 261-282)
#### 2. Attempts to reduce overfitting in the model

I used dropout layers and l2 regularizers to reduce overfitting(model.py lines 261-282). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 154-157). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an RMSprop optimizer, with learning rate of 0.0001 (model.py line 281).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also I drove the car backward for more generalizaiton. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was 
- Data Collection and Exploration
- - Collect Data from simulator and analyze the steering angle distribution 
- - Split the dataset into train and validation set
- Train a model
- - Start from a simple CNN model then increase the convolutional layers 
- - Reduce Overfitting
- - Test on the simulator

***Data Collection***

At first, I collected the dataset using the Simulator . I drove on the normal track at first. I used normal driving , drove into reverse direction , also did recovery driving . Then I drove in difficult track. 

***Data Analysis***

After data collection, I analyzed the dataset. I plotted the histogram of the steering angle distribution. I noticed that most steering angles are near 0 . It might cause bias towards that value. So I removed the steering angles with value 0

***Model Train***

At first I started with lambda layer to normalize the images and cropping layer to crop the relevent portion. This was suggested in the preceding course. Then I trained using one convolutional layer model to get a baseline. It worked moderate on the dataset. But failed on the simulator. Then I started increasing complexity of the model. But It started overfitting. Then I introduced Dropout layers to reduce the overfit. But It didnt work that mucht. So I introduced the l2 regularizer . It decreased overfitting extremely well.



#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Architecture][arch]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center1][normal_center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![normal_recovery1][normal_recovery1]
![normal_recovery2][normal_recovery2]
![normal_recovery3][normal_recovery3]

Then I repeated this process on track two in order to get more data points.



![difficult1][difficult1]
![difficult2][difficult2]

Etc ....

After the collection process, I had 3964 data points for normal track and 4987 data points for track 2 . I then preprocessed this data by normalizing and cropping them while training.


I didnt use augmentation technique as I could not use augmentation while training as the `drive.py` will use the model directly

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 43 as I used `EarlyStopping` method from `Keras` ... I used an RMSprop optimizer with learning rate of `0.0001`.

### Video

I have saved the videos. `video_track1.mp4` is the track1 video. `video_track2.mp4` is for track 2