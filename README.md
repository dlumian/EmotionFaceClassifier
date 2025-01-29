# Emotion Face Classifier

Computer vision project to classify facial expressions into 7 emotion categories: 

Categories are: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

Process includes EDA, image feature extraction (unsupervised), and classification.

## Sections
- [Project Structure](#project-structure)
- [Data Source: FER 2013](#data-source-fer-2013)
- [Model Type Overview](#model-type-overview)
    - [Vectorized Models](#vectorized-models)
    - [Deep-Learning Models](#deep-learning-models)

## Project Structure
[Return to Top](#sections)

- [configs](./configs)
    - input_mappings.json: project details such as emotion-label mapping and plotting styles
    - unsupervised_models.json: model configurement for feature extration
    - vectorized_models.json: model configurement for classification models using vectorized input
- [data](./data/)
    - [Initial data](#data-source-fer-2013) with readme
    - Intermediate data saved here during analysis
- []

## Data Source: FER 2013

[Return to Top](#sections)

Data comes from a Kaggle dataset. Images are 48x48 pixels and in greyscale.

### [Kaggle FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
    - Stored as a string array representing pixel values in a csv file
    - Primary data divided into three usage categories: train, public test, private test
    - Current analyses combines public and private tests
    - Data must be downloaded from Kaggle
    - When uncompressed, move `fer2013.csv` into path: `EmotionFaceClassifier/data/fer2013.csv`

![Image Counts by Category](./imgs/counts/emotion_img_counts_stacked_totals.png)

## Notebooks

- [0_reset_project.ipynb](./notebooks/0_reset_project.ipynb)

    This optional notebook removes existing intermediate data, images, and models but leaves the original csv data, readme, and code in place. Its use accelerates rapid iterations when trying to replicate or expand analysis or when using this repo as a learning tool.

    If unfamiliar with file pattern matching, skip this notebook and remove unwanted files by hand.

- 

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

## To-Do
- environment files
- add logging
