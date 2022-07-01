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



<p align="center">
  <img src="./images/waveforms.png" alt="waveforms" width="550"/>
</p>

### Data Preprocessing

### Model Selection

### Model Training

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


