# Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture

# Introduction
This repository provides the source code and raw datasets of the project "Visual Sentiment Analysis Using Different Neural Network Architecture". Visual sentiment analysis, crucial across psychology, neuroscience, computer vision, and machine learning due to the multifaceted and subjective nature of emotions, varying considerably across individuals, cultures, and contexts. This experiment investigates the effectiveness of fine-tuned VGG architecture-based transfer learning models in addressing the challenges of visual sentiment analysis. Capitalizing on Visual Geometry Group models, specifically VGG16 and VGG19, the project concentrates on sentiment analysis within visual content. Fine-tuning methodologies, including selective layer freezing & unfreezing and regularization techniques, were applied to optimize model performance. In this experiment I have used famous KDEF and CK+ datasets, both comprising seven distinct emotion classes. A comparative analysis of pre-trained models was performed to assess
their performance in predicting image sentiments across the dataset. Notably, the fine-tuned VGG architecture demonstrated tremendous performance compared to conventional transfer learning models. On the KDEF dataset, VGG16 achieved an accuracy of 94.21%, while VGG19 achieved 91.83%, accuracy in test data. Further validation on the CK+ dataset showcased outstanding accuracy, reaching 98.5% for VGG19, whereas VGG16 lagged at 4%.
# Architecture
The evaluation of my visual sentiment analysis model, based on fine-tuned transfer learning from pre-trained VGG-16 and VGG-19 architectures. I chose the VGG because of its clear architecture, featuring repetitive convolutional layer blocks, making it easy to implement. Despite being an older architecture, VGG has notable performance in specific image recognition tasks. However, More intricate models such as ResNet50 or DenseNet121 might face the problem of overfitting issues given my smaller dataset size. Using pre-trained VGG models from ImageNet gives an advantage due to their generalization. In my workflow, initial dataset pre-processing involved employing data augmentation techniques like rotation, shifting, and flipping, alongside histogram equalization to spread out intensity levels in the images. Furthermore, during pre-processing, I resized the images to (224,224,3) to suitable my model’s requirements. The dataset was initially divided into training sets and test sets 80:20 ratio respectively. Following that, an 80:20 split within the training set made distinct training and validation sets, which were utilized for hyperparameter adjustment and model evaluation during training. The One-hot encoder was then used to convert integer labels to binary representations. VGG19 and VGG16 models were imported with pre-trained weights from the ImageNet dataset. The fully connected layers were removed to allow customization of the base models. All layers within the loaded model were frozen to prevent updates during training, except the last four layers. Subsequently, a transfer learning model was built by appending layers at the top of the VGG architectures. This included a Global Average Pooling layer, a Dense layer employing ReLU activation, a Dropout layer with a specified dropout rate, and a final Dense layer with a softmax activation function for the classification of emotions into seven categories. This creates a new model inheriting the base model’s weights while integrating additional layers. Hyperparameter optimization was performed to identify the configuration of the optimal model. I then applied the images to the model and used the results of the model for comparative analysis. Additionally, these models were tested on a new dataset to assess performance in the domain of emotion recognition.
The Overall architechturs that I have used are given below:
![image](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/3c3a143f-9144-4b69-a3f9-86092500eac0)

The Transfrer Learning Models that I have used is given below:

![image](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/a15a0802-440c-45a9-9670-20e1f456ef10)

# Data

**KDEF**
The Karolinska Directed Emotional Faces (KDEF) dataset includes 4900 colored images of human facial emotions. Averaged KDEF (AKDEF) is a collection of averagedimages derived from the original KDEF photos. The KDEF and AKDEF materials
were created in 1998 and have since been freely distributed throughout the scholarly community. KDEF has been used in over 1500 research publications as of today. The KDEF includes seven emotion classes: anger, neutral, disgust, fear, happiness, sadness, and surprise. The complete dataset is available at [here](https://www.kdef.se/).  

**CK+**
The Extended Cohn-Kanade (CK+) dataset consists of 981 grayscale images of seven emotion classes(anger, contempt, disgust, fear, happiness, sadness, and surprise). The CK+ dataset images are (48,48) and grayscale. The CK+ database is widely
regarded as the most extensively used laboratory-controlled facial expression classification dataset available and is used in the majority of facial expression classification methods. The dataset is available [here](https://www.kaggle.com/datasets/shawon10/ckplus).

# How to run the code
The code was run in Jupyter notebook. Please, download the 
```bash
Visual Sentiment Analysis with histogram.ipynb 
```
```bash
Visual Sentiment Analysis without Histogram Equalization.ipynb
```
files from the main directory. Upload the files in the jupyter notebook and run. You must have to download/import the libararies and edit the source folder for the dataset. These code are separated in two files to save the computational power and complexity.

1. You could get the With Histogram + With HyperTuning Results of KDEF from 'Visual Sentiment Analysis with histogram.ipynb '
2. You could get the Without Histogram + Without HyperTuning Results of KDEF from 'Visual Sentiment Analysis without histogram.ipynb '
3. You could get the Without Histogram + with HyperTuning Results of KDEF from 'Visual Sentiment Analysis without histogram.ipynb '
4. You could get the With Histogram + With HyperTuning Results of CK+ from 'CK+ with histogram.ipynb '
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
| learning_rate(adam)   | 0.00001, 0.000001, **0.0001**,0.001,0.01  |
| dropout_rate  | 0.3,0.5,0.7,0.8,**0.1**,0.01,0.001  |
| l2_penalty  | 0.1,**0.01**,0.001 |
| Epochs | 300 |
| Early Stopping | val_loss= 2,3,**4**,5 |
| Number of Layers Unfreezed | 0,1,2,3,**4** |

Note*: The values that are in bold are used in the experiments
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

The models are tested with their test data to see of it can classify the emotions properly.
# System Specification
* Windows 10 Education 64-bit
* Ram: 32 GB
* CPU: Intel Core i7-9700k 3.60 GHz
* GPU: NVIDIA RTX 2080
* Platform : Docker Jupyter Notebook

# Outputs
**Note:** The new output folder will be updated to give clear understadning(updating)

For the overall performance comparison, we have created a table.
![image](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/863c9cb2-cd3d-4e56-80f3-4e13020926ca)

My Hyper-tuned models were able to identify the emotions related to the classes very accurately and some of the predictions are given below:
![image](https://github.com/jaherchowdhury/Visual-Sentiment-Analysis-Using-Different-Neural-Network-Architecture/assets/146418350/97438454-4ff0-41e6-9191-211dd1763327)


