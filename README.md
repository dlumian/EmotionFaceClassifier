# Emotion Face Classifier

Computer vision project to classify facial expressions into one of 6 emotion types: 
- Angry
- Fear
- Happy
- Sad
- Surprise 
- Neutral

**Disgust** is a category in one dataset and is underrepresented there, therefore it will be dropped from analyses.

## Sections
- [Data Sources](#data-sources)
    - [FER 2013](#kaggle-fer2013)
    - [FRD 2020](#kaggle-facial-recognition-dataset)
    - [Data Format Standardization](#data-format-standardization)

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

## To-Do
- environment files
- add logging
