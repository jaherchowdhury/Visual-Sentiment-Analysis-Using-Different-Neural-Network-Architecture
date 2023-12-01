# Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture

# Introduction
This repository provides the source code and raw datasets of the project "Visual Sentiment Analysis Using Different Neural Network Architecture". Visual Sentiment Analysis is the process of detecting and understanding human emotions from visual data, such as images or videos. It involves using computer vision and machine learning techniques to analyze facial expressions, body language, and other visual cues to infer the emotional state of individuals depicted in the images.
Transfer learning is a machine learning technique where a model trained on one task is reused or adapted as a starting point for a different but related task. Instead of training a model from scratch, transfer learning leverages knowledge gained from solving one problem and applies it to a different but related problem.

Now-a-days, Visual sentiment analysis and transfer learning has become a famous topic in computer vision. We are proposing a comparative analysis of the neural network models with the help of transfer learning and how this transfer learning is impacting the field of visual sentiment analysis.We are also proposing how the histogram analysis and different hyperparameters are impacting the models performance.
# Architecture
The Evaluation will be done on two different stages:
![Transfer Learning Algorithms](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/e31716a8-b9d0-4578-bfe4-899c6bcc2aa5)

* The Dataset will be Pre-processed with the Data-AUgmentation( rotation, flipping and re-scaling). Then if the images are not histogram equalized then it will be equalized by the histogram equalization. Fot this, we transform the images to grayscale, then applied the histogram equaliztion and put it back to the orginal images.
* After that the we have used two algorithms run our datasets(vgg16,vgg19).We have used the two version of these algorithms, the normal one and the fine-tuned version for the transfer learning.
* After running alogirthms, we evaluted the algorthms performance by training,validation accuracies. Also, we printed the classification reports and confusion matrix to find out how the model is performing.
*  Then, We evaluated these four models, depending on their accuracies. After that we have chose the best one to go through the K-fold validation to confirm the model performance.
*  Lastly, we gathered the algorithms performance for applying histogram equalization and also for not applying the histogram equalization.The basic comparison results were given in the ```main\output```folder.

# Data

The Karolinska Directed Emotional Faces (KDEF) is a set of totally 4900 pictures of human facial expressions.The KDEF and AKDEF materials were produced in 1998 and have since then been freely shared within the research community. By today, KDEF has been used in more than 1500 research publications. If you wish to find out for which these are, and for what purposes they materials have been used, you can browse the existing publications on google scholar and also a similiar work here(https://www.mdpi.com/2079-9292/10/9/1036). This dataset that we have used in our experiment is the subset of the KDEF dataset and it contains 2940 images(each class contains 420 images). The emotion levels are divided into seven catagories(anger,fear, disgust, happy, sad, surprise, neutral). The sample dataset could be found on the ```main\Data\KDEF```folder.

# How to run the code
The code was run in Jupyter notebook. Please, download the 
```bash
Visual Sentiment Analysis with histogram.ipynb 
```
```bash
Visual Sentiment Analysis without Histogram Equalization.ipynb
```
files from the main directory. Upload the files in the jupyter notebook and run. You must have to download/import the libararies and edit the source folder for the dataset. These code are separated in two files to save the computational power and complexity. One code will give the alogirthms performance with histogram equalization and without the histogram equalization. The final performance of the models with or without the histogram equalization will be found on the 
```bash
Visual Sentiment Analysis without Histogram Equalization.ipynb
```

# Library Dependencies
1. Numpy
2. Pandas
3. OS
4. Keras
5. OpenCV
6. Seaborn
7. Matplotlib
8. Joblib
   
# Models & Layers
In our experminets we used two models VGG19 and VGG16. For the transfer learning, we have used the pre-trained VGG16 and VGG19 which have the weights of the imagenet.(These layers can be used from tensorflow.keras.layers.)Through our project I used different layers like:
- GlobalPooling2D
- Dropout
- Dense <br>

# HyperParameters and Optimizer
We have used Adamm Optimizer for our base models like VGG19, VGG16. For the transfer learning we have used the Adam optimizer.
There are lots of Hyperparameter involved in our models like the dropout layer, learning rate, l2_penalty, epochs, early stopping, Number of Lyaers Unfreezed Layers.
* We have used the Droput,Kernel Regularization, and early-stopping to prevent the overftting.
* Learning rate determines the step size at which the model weights are updated during the training process.
* We have also fine-tuned our models by un-freezing some layers to learn the imagenet weights with our new dataset weights.
* Epochs refer to the number of times the entire dataset is passed forward and backward through the neural network during the training process. We have almost 300 epochs, but it was controlled by the Early Stopping Function.

The list of the hyper-parameter and their values are given in the below table:
| Parameters  | Values |
| ------------- | ------------- |
| learning_rate(adam)   | 0.0001  |
| dropout_rate  | 0.5  |
| l2_penalty  | 0.01 |
| Epochs | 300 |
| Early Stopping | val_loss= 3 |
| Number of Layers Unfreezed | 2 |

# Training & Testing
For Training the dataset we have splitted the dataset into training and testing. We preprocessed the data using data augmentation and histogram equalization.

| Variables  | Meaning | Values |
| ------------- | ------------- | ------------- |
| input_size   | Input size of the Image  | (224,224) |
| batch_size   | Sample of Image Processed by the model at each iteration  | 32 |
| rotation_range  | Rotates  the image to specific number of degrees  | 20 |
| width_shift_range,height_shift_range  | Allows shifting the image horizontally by a maximum of 20% of the image width  | .2 |
| rescale  | Normalizes pixel values to the range [0, 1]   | 1./255 |
| num_classes  | Number of emotion catagories  | 7 |
| K | Number of Folds | 3 |

The models are tested with their test data to see of it can classify the emotions properly. For further assurement, the best model was checked with the K-fold validation.The testing results are given in the output folder.

# System Specification
* Windows 10 Education 64-bit
* Ram: 32 GB
* CPU: Intel Core i7-9700k 3.60 GHz
* GPU: NVIDIA RTX 2080
* Platform : Docker Jupyter Notebook
# Outputs (will be updated)
In the Output folder there are 5 folders of the models performance.In each folder there will be 4 images of the Confusion Matrix, Classification Report, Emotion label detected by the algorithms, Loss and Accuracy over time.They are listed as below:
01. VGG19 with all the layers freezed
02. VGG19 with all the layers freezed accept the last two layers
03. VGG16 with all the layers freezed
04. VGG16 with all the layers freezed accept the last two layers
05. K-fold Validation <br>

For the overall performance comparison, we have created a table. This will be given outside the folders <br>
* Overall Performance

