# MNIST-Number-and-Letter-Detection

Using the MNIST database for number detection, and the EMNIST extension for letter detection. Using tensorflow for machine learning.

For a more direct comparison between digitdetector and letterdetector, I made a new file called newtest.py.
It runs the letterdetector setup if you type "true", and digitdetector setup if you type anything else. Both use the same neural network afterwards.



I potentially found the issue: the image data seems to be in a different format. The imported dataset has values that range from 0 to 255, whereas the working dataset embedded into tensorflow ranges from 0 to 1. The two datasets also print in different formats.
Both datasets print to show the comparison.

Update 5/20:
Was able to trim down the dataset to only the first 10 letters.
Shuffling the training data caused the machine learning to not work for some reason. Changing the shuffling method fixed this.

Update 5/26:
EMNIST letter detection working, use letterdetector.py
