# Speaker Recognition using Spectral Analysis and ML - Design History File
By ADAM HORSLER, RICCARDO EL HASSININ, SEBASTIAN GILLMAN, XIXIAN HUANG and IBRAHIM MOHAMED


</br>

## Table of content
1. [Introduction](./README.md#Introduction)
2. [Project Structure](./README.md#project-structure)
    1. [Gantt Chart](./README.md#gantt-chart)
    2. [Flowchart](./README.md#flowchart)
3. [Exploratory Data Analysis](./README.md#exploratory-data-analysis)
    1. [Problem Formulation](./README.md#problem-formulation)
    2. [Data Aquisition](./README.md#data-acquisition)
    3. [Data Preprocessing](./README.md#data-preprocessing)
    4. [Model Selection](./README.md#model-selection)
    5. [Model Training](./README.md#model-training)
    6. [Final Model Architecture](./README.md#final-model-architecture)
    7. [Model Evaluation](./README.md#model-evaluation)
4. [Inference Algorithm](./README.md#inference-algorithm)
    1. [Database](./README.md#database)
    2. [Algorithm](./README.md#algorithm)
    3. [Testing](./README.md#testing)
5. [Microcontroller Implementation](./README.md#microcontroller-implementation)
    1. [Microcontroller Selection](./README.md#microcontroller-selection)
    2. [Microcontroller Configuration](./README.md#microcontroller-configuration)
    3. [Inference on Microcontroller](./README.md#inference-on-microcontroller)
6. [Optimization](./README.md#optimization)
    1. [Cloud Optimization](./README.md#cloud-optimization)
    2. [Model Optimization](./README.md#model-optimization)
7. [Ethics and Sustainability](./README.md#ethics-and-sustainability)
    1. [Ethics](./README.md#ethics)
    2. [Sustainability](./README.md#sustainability)

</br>

## Introduction

This document serves as the Design History File of Group 16’s project. The goal of the project was to develop a near real-time speaker recognition Machine Learning (ML) product on a microcontroller. The input is a person’s voice, and the output is a prediction of the name of who that speaker is (if they exist in the database). 

The applications of our product are quite open to interpretation by the user. We have developed a speaker recognition platform that could be incorporated into larger applications such as a foundation for a basic voice assistant or a voice enabled security system (either on a small scale with local database or company-wide with a cloud based database). The product has features that enable a versatile number of use cases.

The software is available on this GitHub repository, which can be used to view the complete software development history of the project. The data used for training and testing includes a combination of public datasets and the group’s own. The group’s own dataset can be found here the GitHub in `TrainingDataGen`. The product should be functional on all microcontrollers that meet the minimum specification as outlined in the Hardware section, however development and testing was only conducted on a Raspberry Pi 4, thus this is the recommended microcontroller for guaranteeing reproducible results. 

</br>

## Project Structure

### Gantt Chart
The project involved 7 weeks of development and the group followed the Agile development process, which includes several “sprints” with varying end goals to achieve the final product. 

The Gantt chart below illustrates these sprints and how the project advanced, from start to finish, across the 7 weeks of development. 


### Flowchart
The flowchart below shows how the various components of the product interact with each other. 

<p align="center">
  <img src="./images/Flowchart.png" alt="flowchart"/>
</p>

</br>

## Colab Notebook

### Problem Formulation
Discussions were held between the group and client, who was a representative from ARM. The initial functional and non-functional requirements were given to the group and are outlined in the table below.

|     Functional                                                     |     Non-functional                                                           |
|:------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|     One-word specific   speaker recognition accuracy of 80%        |     Minimize storage and   computation capacity while maximizing accuracy    |
|     Latency of <3 seconds per prediction                           |     Cloud-enabled features                                                   |
|     Microcontroller-isolated   environment for input and output    |     Maximize portability between microcontrollers                            |

### Data Aquisition

The first component of the project involved experimenting with different datasets and understanding the type and quantity of data required for a successful product. The group focused on a combination of public datasets, for example the Speech Commands dataset  and zhvoice corpus , and the group’s own dataset . The format of all speech datasets must be encoded using the Waveform Audio File Format (WAV). This is because WAV files do not lose any information when it comes to frequencies on the sound spectrum. 

The group has decided to combine the zhvoice corpus dataset with our own group dataset, which resulted in a total of 3253 people’s speech data. These audio files add up to around 900 hours in total and are clips of voice samples at maximum of 3 seconds. The different speakers are labelled into different integers, in the range of 0-3252.

As the zhvoice corpus contains audios in MP3 format sampled at 16kHz, we have created a python script `src/create_data.py` that converts all the audios to WAV format. Additionally, the program creates a list containing the file path to each audio and its corresponding classification label (the unique ID of the speaker). This data list is mainly for the convenience of passing each labeled voice through the preprocessing and training process. 

Below are three example of waveforms from the group dataset
<p align="center">
  <img src="./images/waveforms.png" alt="waveforms" width="550"/>
</p>

### Data Preprocessing
With the data list created, the spectrogram of each audio can be obtained. A spectrogram is a visual representation of the change on distribution of energy among different frequencies over time.  They contain rich information as their various shape displayed reveal the features of voice. For example, they can be used to distinguish the natural frequencies of people's tones. 

The preprocessing of the WAV audios has multiple stages to make it more suitable and faster to process for the ML model. The waveforms are converted into amplitude spectrograms using Short Time Fourier Transforms (STFT). It was discussed whether to use STFT, Mel-frequency spectrograms or Mel-frequency cepstral coefficients spectrograms as they are more used in speech recognition, but after a first round of testing it was decided that linear-frequency spectrograms were satisfactory enough to extract the voiceprint of the speakers. 

The APIs used for this task are `librosa.stft()` and `librosa.magphase()`. This can be shown in the python script `src/utils/reader.py`.

During training, data augmentation, such as random flip stitching, random cropping, frequency masking, is used. After processing the data, spectrograms with the shape of 257x257 are obtained. 

Finally, the spectrograms are split into training and test sets, with a 90:10 ratio. Two example spectrograms are shown in the figures below:

<p float="left">
  <img src="./images/Seb.png" width="400" />
  <img src="./images/Riccardo_linear.png" width="400" /> 
</p>

The specification of preprocessing data samples is listed in the table below:

| Argument                      | Value        |
|-------------------------------|--------------|
|     Sample   Rate             |     16000    |
|     Frame   length            |     0.025    |
|     Window   length           |     400      |
|     Frame   stride            |     0.01     |
|     Hop   length              |     160      |
|     Spectrogram   length      |     257      |
|     Length   of FFT points    |     512      |

### Model Selection

### Model Training
For the Resnet model, after processing the data into 257x257 spectrograms, we fed them into our CNN network. As seen in `src/train.py`, the data input layer is [None,1,257,257] which matches the shape of spectrograms. For training, we use a stochastic gradient descent optimizer with a learning rate is 0.001 and the number of epochs is set to be 50. The loss function chosen is Additive Angular Margin Loss (ArcFace Loss). This loss function is used to normalize the features vector and weights, making the predictions only depend on the angle between the feature and the weight where an additive angular margin penalty m is added to θ (angle between weights and the features)

<p align="center">
  <img src="./images/loss_function.png" alt="loss"/>
</p>

Overall, the ArcFace loss helps the model to maximize the margin which is the decision boundary on the hyperplane. It obtains discriminative features for speaker recognition and helps the model to calculate the geodesic distance between features on the hyperplane.

A dense layer with 3252 classes was added at the end of the ResNet5o model, in order to classify the 3253 people in the dataset during training. The following pictures showcase the training accuracy and loss of the model over 50 epochs.

<p float="left">
  <img src="./images/accuracy.JPG" />
  <img src="./images/loss.JPG" /> 
</p>

### Final Model Architecture 

### Model Evaluation

</br>

## Inference Algorithm

### Database

### Algorithm

### Testing

</br>

## Microcontroller Implementation

### Microcontroller Selection

### Microcontroller Configuration

### Inference on Microcontroller 

</br>

## Optimization

### Hardware Optimization

### Model Optimization

</br>

## Ethics and Sustainability

### Ethics

### Sustainability


