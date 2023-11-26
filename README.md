# Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture
#Introduction
This repository provides the source code and raw datasets of the project "Visual Sentiment Analysis Using Different Neural Network Architecture".Now-a-days, sentiment analysis and transfer learning has become a famous topic in computer vision. We are proposing a comparative analysis of the neural network models with the help of transfer learning and how this transfer learning is impacting the field of visual sentiment analysis.We are also proposing how the histogram analysis and different hyperparameters are impacting the models performance.



# Architecture
The Evaluation will be done on two different stages:
![Transfer Learning Algorithms](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/e31716a8-b9d0-4578-bfe4-899c6bcc2aa5)
![Transfer Learned Mode](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/a57ac5e7-9046-4c24-838f-15a689ec0e60)

# Data

The Karolinska Directed Emotional Faces (KDEF) is a set of totally 4900 pictures of human facial expressions.The KDEF and AKDEF materials were produced in 1998 and have since then been freely shared within the research community. By today, KDEF has been used in more than 1500 research publications. If you wish to find out for which these are, and for what purposes they materials have been used, you can browse the existing publications on google scholar and also a similiar work here(https://www.mdpi.com/2079-9292/10/9/1036).


# Library Dependencies
1. Numpy
2. Pandas
3. OS
4. Keras
5. OpenCV
6. Seaborn
7. Matplotlib
8. Joblib
   
# Models
In our experminets we used three models(VGG19,ResNet50,DenseNet121). All the models are availble in the tf.keras.applications.
# Layers
Through our project I used different layers like:
* Flatten
* BatchNormalization
* GlobalPooling2D
* Dropout
* Dense
These layers can be used from tensorflow.keras.layers.

# HyperParameters and Optimizer
We have used Adamm Optimizer for our base models like VGG19 or ResNet50. For the transfer learning we have used the SGD(Stochastic Gradient Descent) optimizer.
There are lots of Hyperparameter involved in our models. The list of the hyper-parameter and their values are given in the below table:

| Parameters  | Values |
| ------------- | ------------- |
| learning_rate(adam)   | 0.0001  |
| learning_rate(sgd)   | 0.0001 |
| momentum  | 0.9  |
| dropout_rate  | 0.5  |
| l2_penalty  | 0.01 |

# Training
For Training the dataset we have splitted the dataset into we have took the images of train folder. We preprocessed the data using data augmentation and histogram equalization.

| Variables  | Meaning | Values |
| ------------- | ------------- | ------------- |
| input_size   | Input size of the Image  | (48,48) |
| batch_size   | Sample of Image Processed by the model at each iteration  | 64 |
| rotation_range  | Rotates  the image to specific number of degrees  | 20 |
| width_shift_range,height_shift_range  | Allows shifting the image horizontally by a maximum of 20% of the image width  | .2 |
| rescale  | Normalizes pixel values to the range [0, 1]   | 1./255 |
| num_classes  | Number of emotion catagories  | 7 |



