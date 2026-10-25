# Future work: a learned emotion model

The current emotion model is analytic (hand-authored modulation of gait curves). This note preserves the original plan for a learned replacement, transcribed from the retired `emotionmodel.py` stub.

## Candidate datasets

- EWalk (happy, sad, angry, neutral)
- Kinematic dataset of actors expressing emotions (all basic emotions)
- Emotion-Gait (happy, sad, angry, neutral)
- Human3.6M
- CMU Mocap
- ICT
- BML
- SIG

## Plan

1. Find datasets to train on (the shortlist above).
2. Find an existing model that takes in a sequence of joint angles and outputs an emotion prediction.
3. Train the model on the dataset, given desired joint angles and an emotion, with a loss that combines the emotion prediction and the deviation from the original joint angles.

## Emotion labels

The stub used this index map: anger 0, disgust 1, fear 2, happiness 3, sadness 4, surprise 5, neutral 6.

## Integration seam

The analytic emotion model in `emotion/` is the seam a learned model would replace. The function to swap is `emotion.model.modulate`: a learned model would provide the same interface, taking base joint angles and emotion intensities and returning modulated joint angles.
