#import statements
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
import copy
import os
import idx2numpy

#import data files
trainLabels = idx2numpy.convert_from_file('emnist-letters-train-labels-idx1-ubyte.txt')
trainImages = idx2numpy.convert_from_file('emnist-letters-train-images-idx3-ubyte.txt')
testLabels = idx2numpy.convert_from_file('emnist-letters-test-labels-idx1-ubyte.txt')
testImages = idx2numpy.convert_from_file('emnist-letters-test-images-idx3-ubyte.txt')

#shuffle the data
trP = np.random.permutation(len(trainImages))
trainImages = trainImages[trP]
trainLabels = trainLabels[trP]
teP = np.random.permutation(len(testImages))
testImages = trainImages[teP]
testLabels = trainLabels[teP]

#images are stored column by column for some reason, this fixes that.
for i in range(0, len(trainImages)):
    trainImages[i] = trainImages[i].transpose()
for i in range(0, len(testImages)):
    testImages[i] = testImages[i].transpose()


#examples from both training and testing datasets
for i in range(0, 10):
    print("Training set examples: {} ({})".format(chr(trainLabels[i]+64), trainLabels[i]))
    plt.imshow(np.asarray(trainImages[i]).reshape(28,28), cmap='gray')
    plt.show()
for i in range(0, 10):
    print("Test set examples: {} ({})".format(chr(testLabels[i]+64), testLabels[i]))
    plt.imshow(np.asarray(testImages[i]).reshape(28,28), cmap='gray')
    plt.show()

#matplotlib event listener
loop = True
def press(event):
    if event.key == 'q':
        global loop
        loop = False
    if event.key == ' ':
        plt.close()

trainLabelsTemp = np.zeros(3244800).reshape(124800,26)
testLabelsTemp = np.zeros(540800).reshape(20800,26)

for i in range(0, len(trainLabels)):
    value = int(trainLabels[i])-1
    trainLabelsTemp[i][value] = 1

for i in range(0, len(testLabels)):
    value = int(testLabels[i])-1
    testLabelsTemp[i][value] = 1

trainLabels = trainLabelsTemp
testLabels = testLabelsTemp
'''
for i in range(0,20):
    print(trainLabels[i])
    print(np.argmax(trainLabels[i]))
    plt.imshow(trainImages[i].reshape(28,28), cmap='gray')
    plt.show()

for i in range(0,20):
    print(testLabels[i])
    print(np.argmax(testLabelsTemp[i]))
    plt.imshow(testImages[i].reshape(28,28), cmap='gray')
    plt.show()
'''
#neural network setup
X = tf.placeholder(tf.float32, shape=(None, 784), name='X')
labels = tf.placeholder(tf.float32, shape=(None, 26), name='Labels')

#first round
w1 = tf.Variable(name='w1', initial_value=tf.random_normal(shape=(784, 128), dtype=tf.float32))
b1 = tf.Variable(name='b1', initial_value=tf.zeros(shape=(1,128), dtype=tf.float32))
a1 = tf.matmul(X, w1) + b1
z1 = tf.nn.relu(a1)

#second round
w2 = tf.Variable(name='w2', initial_value=tf.random_normal(shape=(128, 32), dtype=tf.float32))
b2 = tf.Variable(name='b2', initial_value=tf.zeros(shape=(1,32), dtype=tf.float32))
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.relu(a2)

#third round
w3 = tf.Variable(name='w3', initial_value=tf.random_normal(shape=(32, 26), dtype=tf.float32))
b3 = tf.Variable(name='b3', initial_value=tf.zeros(shape=(1,26), dtype=tf.float32))
y = tf.matmul(z2, w3) + b3

#training setup and parameters
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=y)
step = tf.train.AdamOptimizer().minimize(loss)
BATCH_SIZE = 32
EPOCHS = 10
m = 124800

#creates training session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training followed by testing every 5 epochs
for epoch in range(EPOCHS):
    for i in range(BATCH_SIZE, m, BATCH_SIZE):
        trainBatch = (trainImages.reshape(124800,784))[i-BATCH_SIZE:i]
        labelBatch = (trainLabels.reshape(124800,26))[i-BATCH_SIZE:i]
        sess.run(step, feed_dict={X:trainBatch, labels:labelBatch})
    #if (epoch+1) % 5 = 0:
    predictions, test_loss = sess.run([y, loss], feed_dict={X:testImages.reshape(20800,784), labels:testLabels.reshape(20800,26)})
    accuracy = np.mean(np.argmax(testLabels.reshape(20800,26), axis=1) == np.argmax(predictions, axis=1))
    print('Epoch: %d\tloss: %1.4f\taccuracy: %1.4f' % (epoch+1, test_loss, accuracy))

#allows for saving and loading models. either run restore or save
saver = tf.train.Saver()
#saver.restore(sess, "mnist_nn.txt")
saver.save(sess, "lmnist_nn.txt")

'''
#appends user-created images and their labels to the lists
testImages = np.insert(testImages, 0, newImage, axis=0)
testImages = np.insert(testImages, 0, np.asarray(createdImages), axis=0)
testLabels = np.insert(testLabels, 0, np.asarray([0,0,1,0,0,0,0,0,0,0]), axis=0)
testLabels = np.insert(testLabels, 0, np.asarray(createdLabels), axis=0)
'''

#runs the neural network on the images
predictions = sess.run(y, feed_dict={X:testImages.reshape(20800,784)})
'''
for i in range(0, len(predictions)):
    print(testLabels[i])
    print(predictions[i])
'''
#variables for viewing loop
correct = 0
total = 0
loop = True
mplcount = 0

#matplotlib eventlistener
def press(event):
    if event.key == 'q':
        global loop
        loop = False
    if event.key == ' ':
        plt.close()

#while loop displays and image and prints the value and the neural network's guess. spacebar advances to the next image and q exits the loop
while loop == True and mplcount < len(testLabels):
    print("Prediction: {}, Label: {}".format(np.argmax(predictions[mplcount]), np.argmax(testLabels[mplcount])))
    '''
    if mplcount < count:
        name = "Drawn Image #{}:".format(mplcount+1)
    elif mplcount == count:
        name = "Raw Data 2:"
    else:
        name = "MNIST Image #{}:".format(mplcount-count)
    '''

    plt.figure(figsize=[8.8,4]).canvas.set_window_title("Image #" + str(total) + ": Press space to view the next image, Q to exit")
    plt.subplot(1, 2, 1)
    plt.imshow(testImages[mplcount].reshape(28, 28), cmap='gray')
    x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    y = predictions[mplcount]
    plt.subplot(1, 2, 2)
    if np.argmax(predictions[mplcount]) == np.argmax(testLabels[mplcount]):
        plt.plot(x, y, color='blue', mfc='green', mec='black', marker='o', linewidth=2, markersize=10)
    else:
        plt.plot(x, y, color='blue', mfc='red', mec='black', marker='o', linewidth=2, markersize=10)

    plt.connect("key_press_event", press)
    plt.show()

    if np.argmax(predictions[mplcount]) == np.argmax(testLabels[mplcount]):
        correct += 1
    total += 1
    mplcount += 1

#statistics for the visualization
print("On the test set, the neural network guessed {} correct out of {} total. ({}%)".format(correct, total, round(100*correct/total, 1)))
