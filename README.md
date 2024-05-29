# EmotionFaceClassifier

Computer vision project to determine emotion of a face.

This repo is a revisitation of my [emotion_face_classification](https://github.com/dlumian/emotion_face_classification) repo from 2018.

## Data Sources

Data comes from two Kaggle datasets on emotional face recognition. Overlapping similarities of the two datasets are listed here. Unique features are below each data link. Data can be downloaded from the links and then saved into the structure specified below. Different organization is due to variation in how original data was stored. 
- Emotions: Surprise, Angry, Happiness, Sad, Neutral, Disgust, Fear
- 48x48, greyscale images 


- [Kaggle FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
    - Stored as pixel values in a csv file
    - Path: `EmotionFaceClassifier/data/fer2013/fer2013.csv`
- [Kaggle Facial Recognition Dataset](https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset/data)
    - Stored as jpg files
    - Data is organized into training and testing with a directory for each emotion in those directories. The `Emotion` at the end of the path must be replaced with the proper emotion label.
        - Train path: `EmotionFaceClassifier/data/frd2020/Training/Emotion`
        - Test path: `EmotionFaceClassifier/data/frd2020/Testing/Emotion`



