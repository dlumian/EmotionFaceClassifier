# Emotion Face Classifier

Computer vision project to classify facial expressions into one of 6 emotion types: 
- Angry
- Fear
- Happy
- Sad
- Surprise 
- Neutral

**Disgust** is represented in only one dataset and is underrepresented there. Therefore, it is dropped from analyses.

## Sections
- [Data Sources](#data-sources)
    - [FER 2013](#kaggle-fer2013)
    - [FRD 2020](#kaggle-facial-recognition-dataset)
    - [Data Format Standardization](#data-format-standardization)
- [Model Type Overview](#model-type-overview)
    - [Vectorized Models](#vectorized-models)\
    - [Deep-Learning Models](#deep-learning-models)
-[Reset Project](#reset-project)

## Data Sources

[Return to Top](#sections)

Data comes from two Kaggle datasets. Both use 48x48 pixel, greyscale images.

Unique dataset features are outlined below the link to the data. Data can be downloaded from the links and then saved into the structure specified below. 

### [Kaggle FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
    - Stored as pixel values in a csv file
    - Includes `Disgust` as additional category
    - Has 3 usage categories: train, public test, private test
    - Since not in contest, this project usage combines public and private tests
    - When download is uncompressed, move `fer2013.csv` into path: `EmotionFaceClassifier/data/fer2013/fer2013.csv`

### [Kaggle Facial Recognition Dataset](https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset/data)
    - Stored as jpg files
    - Data is organized into training and testing with a directory for each emotion within those directories
    - The `Emotion` at the end of the path must be replaced with the proper emotion label.
        - Train path: `EmotionFaceClassifier/data/frd2020/Training/Emotion`
        - Test path: `EmotionFaceClassifier/data/frd2020/Testing/Emotion`
    - Of note, `Surprise` is misspelled in the FRD2020 dataset

### Data Format Standardization
As part of the EDA notebook, steps are applied to facilitate analysis. Specifically, data format standardization is applied to the FER2013 dataset to convert pixel arrays from the CSV into image files. By harmonizing both datasets, a consistent data format and structure is ensured for subsequent analysis. 

## Model Type Overview
Two primary modeling approaches are explored: vectorized and deep-learning approaches. In short,vectorized models are simpler and treat images as flat arrays of pixels, making them easier to use but less effective for capturing the inherent structure of image data.
Advanced models like CNNs maintain the 2D structure of images and leverage this to extract complex features, making them much more effective for most image-related tasks but at the cost of increased complexity and resource requirements.

NOTE: Given the difference of computational resources needed, be conscientious of how many and what type of models you are testing. 

### Vectorized Models
Vectorized models flatten 2D image data into 1D vectors. Therefore, a 48x48 pixel image (which has 2,304 pixels) is converted into a single vector of 2,304 elements. Each pixel is treated as an independent feature, so no consideration of the spatial relationships between pixels is taken into account.
The primary advantages of this approach is simplicity of implementation and less computational resources needed compared to more complex models. The primary disadvantage is that vectorized models may perform worse than their more complex alternatives.

### Deep-Learning Models  
Advanced models like Convolutional Neural Networks (CNNs) keep the 2D structure of the image intact. Instead of flattening the image, these models process the image in its original form, taking into account the height, width, and depth (channels) of the image.
These models are capable of automatically learning spatial hierarchies of features through layers of convolutional filters, pooling, and fully connected layers. They can capture edges, textures, shapes, and more complex features as the network deepens.
These models excel at complex image classification tasks due to their ability to understand and utilize the spatial relationships between pixels. 
The primary advantages of deep-learning models is improved accuracy and state-of-the-art performance due to leveraging spatial relationships of the image. 

## Reset Project
Script to be used to clean out intermediate data and results so repo can be run fresh. Removed directories includes `data/intermediate_data`, `models`, `imgs`, and `metrics`. Also resets all notebooks in `notebook` dir. Example usage from utils dir: `python reset_project.py`.

## To-Do
- environment files
- add logging
