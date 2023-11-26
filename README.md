# Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture
#Introduction
This repository provides the source code and raw datasets of the project "Visual Sentiment Analysis Using Different Neural Network Architecture".Now-a-days, sentiment analysis and transfer learning has become a famous topic in computer vision. We are proposing a comparative analysis of the neural network models with the help of transfer learning and how this transfer learning is impacting the field of visual sentiment analysis.We are also proposing how the histogram analysis and different hyperparameters are impacting the models performance.



# Architecture
The Evaluation will be done on two different stages:
![Transfer Learning Algorithms](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/e31716a8-b9d0-4578-bfe4-899c6bcc2aa5)
![Transfer Learned Mode](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/a57ac5e7-9046-4c24-838f-15a689ec0e60)

# Data

The Karolinska Directed Emotional Faces (KDEF) is a set of totally 4900 pictures of human facial expressions.The KDEF and AKDEF materials were produced in 1998 and have since then been freely shared within the research community. By today, KDEF has been used in more than 1500 research publications. If you wish to find out for which these are, and for what purposes they materials have been used, you can browse the existing publications on google scholar and also a similiar work here(https://www.mdpi.com/2079-9292/10/9/1036). This dataset that we have used in our experiment is the subset of the KDEF dataset and it contains 2940 images(each class contains 420 images). The emotion levels are divided into seven catagories(anger,fear, disgust, happy, sad, surprise, neutral)


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
In our experminets we used three models(VGG19, VGG16, ImagenetV2, InceptionV3). All the models are availble in the tf.keras.applications.
# Layers
Through our project I used different layers like:
* Flatten
* BatchNormalization
* GlobalPooling2D
* Dropout
* Dense
These layers can be used from tensorflow.keras.layers.

# HyperParameters and Optimizer
We have used Adamm Optimizer for our base models like VGG19, VGG16, ImagenetV2, InceptionV3. For the transfer learning we have used the SGD(Stochastic Gradient Descent) optimizer.
There are lots of Hyperparameter involved in our models. The list of the hyper-parameter and their values are given in the below table:

| Parameters  | Values |
| ------------- | ------------- |
| learning_rate(adam)   | 0.0001  |
| learning_rate(sgd)   | 0.0001 |
| momentum  | 0.9  |
| dropout_rate  | 0.3  |
| l2_penalty  | 0.01 |
| Epochs | 100 |
| Early Stopping | val_loss=5 |

# Training
For Training the dataset we have splitted the dataset into we have took the images of train folder. We preprocessed the data using data augmentation and histogram equalization.

| Variables  | Meaning | Values |
| ------------- | ------------- | ------------- |
| input_size   | Input size of the Image  | (224,224) |
| batch_size   | Sample of Image Processed by the model at each iteration  | 64 |
| rotation_range  | Rotates  the image to specific number of degrees  | 20 |
| width_shift_range,height_shift_range  | Allows shifting the image horizontally by a maximum of 20% of the image width  | .2 |
| rescale  | Normalizes pixel values to the range [0, 1]   | 1./255 |
| num_classes  | Number of emotion catagories  | 7 |
| K | 2 |

# Testing
The models are tested with their tested data and with a unseen samle image to see of it can classify the emotions properly. For further assurement, the best model was checked with the K-fold validation.
The testing results are given in the output folder.

# Hyper Parameter Tuning
The models are tuned with the hyper-meter. There were several of hyper-meters for this models like, learning-rate, dropout, l2_regularization, etc. We took a group of parameter to test that which set of parameter is giving us the best results.
# Comparison
We evaluated the model by the test and train accuracy. Also, a classification report was done to show the accuracy,f1-score,precision and recall.
Moreover, freezing and unfreezing the layers impacts on the accuracy of the models. That's why I took several group of layers settings and then applied with the hyper-tuned models to find the results.
Lastly, these models was comapred against their accuracies.The basic comparison results were given in the output folder.
