import bvhtoolbox
import os

'''
* Find datasets to train on
    * EWalk (happy, sad, angry, neutral)
    * Kinematic dataset of actors expressing emotions (all basic emotions)
    * Emotion-Gait (happy, sad, angry, neutral)
    * Human3.6M
    * CMU Mocap
    * ICT
    * BML
    * SIG
* Find an existing model that takes in a sequence of joint angles and outputs an emotion prediction
* Train the model on the dataset, giving desired joint angles and emotion with loss function including the emotion prediction and deviation from the original joint angles
'''

emotion_map = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happiness': 3,
    'sadness': 4,
    'surprise': 5,
    'neutral': 6
}

def gather_kiematic_dataset():
    for root, dirs in os.walk('BVH'):
        files = [os.path.join(root, f) for f in dirs if f.endswith('.bvh')]
        for f in files:
            bvh = bvhtoolbox.Bvh(f)
            for frame in bvh.frames:
                print(frame)
                break
            break

def gather_ewalk_dataset():
    pass

def gather_emotion_gait_dataset():
    pass