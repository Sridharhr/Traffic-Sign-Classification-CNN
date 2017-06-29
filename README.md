# dl-traffic-sign

### About

Use of convolutional neural networks to classify traffic signs. Trains a model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). 

### Data

* [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* The code uses a pickled (i.e. binary serialized) version the dataset - you can work directly with the dataset at the provided link
* See `signnames.csv` for the label meanings

### Code

`code.ipynb` (Jupyter notebook)

[//]: # (Image References)

[image1]: ./images/histograms.png "Class distribution"
[image2]: ./images/all_classes.png "Sample visualization by class"
[image3]: ./images/balanced.png "Training set distribution after balancing"
[image4]: ./images/gray.png "Grayscaling impact"
[image5]: ./images/20kmph.jpg "Traffic Sign 1"
[image6]: ./images/no_entry.jpg "Traffic Sign 2"
[image7]: ./images/stop.jpg "Traffic Sign 3"
[image8]: ./images/children_crossing.jpg "Traffic Sign 4"
[image9]: ./images/right_turn.jpg "Traffic Sign 5"
[image10]: ./images/dist1.png " "
[image11]: ./images/dist2.png " "
[image12]: ./images/dist3.png " "

The histograms of class distributions indicate many classes that are significantly under-represented. (The distribution is consistent across training, validation and test sets.)

![alt text][image1]

The images are generally of limited quality with variation in color/lighting. Some categories show inherent challenges in classification (e.g. several signs have the general structure of a triangle with a low-resolution 'black blob' in the center).

![alt text][image2]

Without balancing under-represented classes, the accuracy in the training tops out to about 90%.

Data is augmented by replicating images for under-represented classes. The revised distribution is more balanced that the original, and is used for the network training.

![alt text][image3]

Grayscaling improves image features for training - especially for darker images. Though the tradeoff is that color (e.g. red border) is no longer utilized in training the network.

![alt text][image4]

The new data set - after augmentation - consists of 42329 normalized and grayscaledsamples, of shape (32,32,1)


### Model architecture

The model used here consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28X28X6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14X14X6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10X10X16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5X5X16 				|
| Flatten	      	| outputs 400 			
| Fully connected		| 400X120        									|
| RELU		| 					|
| Fully connected		| 120X84        									|
| RELU		| 					|
| Fully connected		| 84X43			| Softmax 



Final model results are:

* validation set accuracy of 0.93
* test set accuracy of 0.911

In getting to this result, network changes had less impact compared to data preprocessing. The major improvements came from training set re-balancing, normalization and grayscaling. Finally only the hyper-parameters needed tuning to get to 93%. Changes made to the network structure did not yield major improvements without the data pre-processing.
 

### Testing

Other than the test set results, five German traffic signs are picked from the web:

![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

Predictions for these were as follows:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 							| 
| No entry     			| No entry							|
| Stop					| Stop											|
| Children crossing	      		| Slippery Road
| Right turn			| Right turn    		
| 20kmph      		| Stop   		

So the model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. Accounting for discretization (since there are only 5 images), the model could have achieved either 80% or 100% accuracy. Compared to the test set accuracy of 91%, (and validation set accuracy of 93%) this suggests the model may have overfit to the provided data.  The dataset, on average, has much lower quality when compared to the internet images. So essentially the model has fit itself to these lower quality images which could lead to misclassification as follows:

* 'Children crossing' and 'Slippery road' both are enclosed in a triangle, with a blurred black image or 'blob' in the center. This makes it difficult for the network to distinguish between the two.

* '20kmph' has a circle boundary and 'Stop' has an octagonal boundary - which at low resolution looks like a circle. Both have text in the middle. Additionally, the original data had 20kmph (class 0) significantly under-represented in the data set (180 samples), whereas 'Stop' (class 14) had 690 samples. This would make the network better at recognizing variations of 'Stop'. Even though we increased the samples of class 0, we did it by replicating the same images - whereas seems the 'Stop' samples had better variation represented in the data set. This suggests repeating the same images in case of class 0 did not improve robustness of the network adequately to recognize that class.


### softmax probabilities for each prediction

The top 5 probability distribution of the class predictions is shown below. It indicates that even when the predictions are incorrect, the network is confident in its predictions. 

This makes it difficult to assess from the 'spread' of the predictions that the network may be making a wrong prediction. This suggests that the reasons for misclassification are inherent in the input data set - for example several signs looking similar in boundary and shape of their enclosed image - hence better pre-processing of the images or higher resolution are needed to avoid misclassification.

![alt text][image10]
![alt text][image11]
![alt text][image12]


