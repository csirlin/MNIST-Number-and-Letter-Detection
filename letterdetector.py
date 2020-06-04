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
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import struct

#file loading
trainImages = idx2numpy.convert_from_file('emnistTrainImages.txt')
testImages = idx2numpy.convert_from_file('emnistTestImages.txt')
trainLabels = idx2numpy.convert_from_file('emnistTrainLabels.txt')
testLabels = idx2numpy.convert_from_file('emnistTestLabels.txt')

#shuffling the data
p = np.random.permutation(len(trainLabels))
trainImages = np.copy(trainImages[p])
trainLabels = np.copy(trainLabels[p])
p = np.random.permutation(len(testLabels))
testImages = np.copy(testImages[p])
testLabels = np.copy(testLabels[p])

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
b3 = tf.Variable(name='b3', initial_value=tf.zeros(shape=(1, 26), dtype=tf.float32))
y = tf.matmul(z2, w3) + b3

#training setup and parameters
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=y)
step = tf.train.AdamOptimizer().minimize(loss)
BATCH_SIZE = 128
EPOCHS = 140
m = len(trainImages)

#creates training session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#runs training data through neural network
'''
#training followed by testing every 5 epochs
for epoch in range(EPOCHS):
    for i in range(BATCH_SIZE, m, BATCH_SIZE):
        trainBatch = trainImages[i-BATCH_SIZE:i]
        labelBatch = trainLabels[i-BATCH_SIZE:i]
        sess.run(step, feed_dict={X:trainBatch, labels:labelBatch})
    if (epoch+1) % 5 < 5: # == 0
        predictions, test_loss = sess.run([y, loss], feed_dict={X:testImages, labels:testLabels})
        accuracy = np.mean(np.argmax(testLabels, axis=1) == np.argmax(predictions, axis=1))
        print('Epoch: %d\tloss: %1.4f\taccuracy: %1.4f' % (epoch+1, test_loss, accuracy))
'''

#allows for saving and loading models. either run restore or save
saver = tf.train.Saver()
saver.restore(sess, "emnist_nn.txt")
#saver.save(sess, "emnist_nn.txt")

#loads computer font letters into the testing data
for i in range(25,-1,-1):
    img = mpimg.imread('Letters/{}.png'.format(chr(i+65)))
    img2=[]
    for j in range(0,28):
        for k in range(0,28):
            img2.append(img[j][k][0])

    testImages = np.insert(testImages, 0, np.asarray(img2).reshape(28,28).transpose().reshape(784), axis=0)

    weight = np.zeros(26)
    weight[i] = 1
    testLabels = np.insert(testLabels, 0, weight, axis=0)

#runs the trained neural network on the testing dataset
predictions = sess.run(y, feed_dict={X:testImages})

#variables for viewing loop
count = 0
correct = 0
loop = True

#matplotlib eventlistener
def press(event):
    if event.key == 'q':
        global loop
        loop = False
    if event.key == ' ':
        plt.close()

#while loop displays an image and shows the strength of the neural network's guesses. spacebar advances to the next image and q exits the loop
while loop == True and count < len(testLabels):

    #printing info in console
    print("Prediction: {}({}), Label: {}({})".format( chr(np.argmax(predictions[count])+65), np.argmax(predictions[count]), chr(np.argmax(testLabels[count])+65),np.argmax(testLabels[count])))

    #naming and sizing window
    name = ""
    if count < 26:
        name = "Computer Font Image #{}. Press space to move on, or Q to quit.".format(count+1)
    else:
        name = "EMNIST Image #{}. Press space to move on, or Q to quit.".format(count-25)

    plt.figure(figsize=[13.2,4]).canvas.set_window_title(name)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    #showing the image
    plt.subplot(1, 3, 1)
    plt.imshow(testImages[count].reshape(28, 28).transpose(), cmap='gray')

    #showing the strength of the guess for each letter
    x = []
    for i in range(0, 26):
        x.append(i)
    y = predictions[count]
    plt.subplot(1, 3, (2,3))
    if np.argmax(predictions[count]) == np.argmax(testLabels[count]):
        plt.plot(x, y, color='blue', mfc='green', mec='black', marker='o', linewidth=2, markersize=10)
    else:
        plt.plot(x, y, color='blue', mfc='red', mec='black', marker='o', linewidth=2, markersize=10)

    #connecting the even listener
    plt.connect("key_press_event", press)
    plt.show()

    #advancing counters
    if np.argmax(predictions[count]) == np.argmax(testLabels[count]):
        correct += 1
    count += 1

#statistics for the visualization
print("On this test set, the neural network guessed {} letters correctly out of {} letters total. ({}%)".format(correct, count, round(100*correct/count, 1)))
