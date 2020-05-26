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

print("Running EMNIST")
print("What number of letters would you like to include?")
lettercount = int(input())

'''
trainLabels = np.copy(idx2numpy.convert_from_file('train-labels-idx1-ubyte'))
trainImages = np.copy(idx2numpy.convert_from_file('train-images-idx3-ubyte'))
testLabels = np.copy(idx2numpy.convert_from_file('t10k-labels-idx1-ubyte'))
testImages = np.copy(idx2numpy.convert_from_file('t10k-images-idx3-ubyte'))
trainImages = trainImages.reshape(60000, 784)
testImages = testImages.reshape(10000, 784)

trainImages = trainImages.reshape(60000, 784)
testImages = testImages.reshape(10000, 784)

'''
with open('emnist-letters-train-images-idx3-ubyte.txt','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    trainImages = data.reshape((size, nrows*ncols)).astype(float)

with open('emnist-letters-test-images-idx3-ubyte.txt','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    testImages = data.reshape((size, nrows*ncols)).astype(float)

with open('emnist-letters-train-labels-idx1-ubyte.txt','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    trainLabels = data.reshape(size).astype(float)

with open('emnist-letters-test-labels-idx1-ubyte.txt','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    testLabels = data.reshape(size).astype(float)

p = np.random.permutation(len(trainLabels))
trainImages = np.copy(trainImages[p])
trainLabels = np.copy(trainLabels[p])

p = np.random.permutation(len(testLabels))
testImages = np.copy(testImages[p])
testLabels = np.copy(testLabels[p])

trainLabelsTemp = np.zeros((4800*lettercount, lettercount))
testLabelsTemp = np.zeros((800*lettercount, lettercount))
trainImagesTemp = np.zeros((4800*lettercount, 784))
testImagesTemp = np.zeros((800*lettercount, 784))

j=0
for i in range(0, len(trainLabels)):
    if trainLabels[i] <= lettercount:
        value = trainLabels[i]
        trainLabelsTemp[j][int(value)-1] = 1.0
        print("TRAINLABEL:{}".format(trainLabelsTemp[j]))
        trainImagesTemp[j] = trainImages[i]
        j+=1
j=0
for i in range(0, len(testLabels)):
    if testLabels[i] <= lettercount:
        value = testLabels[i]
        testLabelsTemp[j][int(value)-1] = 1.0
        print("TESTLABEL:{}".format(testLabelsTemp[j]))
        testImagesTemp[j] = testImages[i]
        j+=1

trainLabels = np.copy(trainLabelsTemp)
testLabels = np.copy(testLabelsTemp)
trainImages = np.copy(trainImagesTemp)
testImages = np.copy(testImagesTemp)

for i in range(0, len(trainImages)):
    for j in range(0, len(trainImages[0])):
        value = trainImages[i][j]
        trainImages[i][j] = value/256.0
    if i%1000 == 0:
        print(str(i))

for i in range(0, len(testImages)):
    for j in range(0, len(testImages[0])):
        value = testImages[i][j]
        testImages[i][j] = value/256.0
    if i%1000 == 0:
        print(str(i))

loop = True
def press(event):
    if event.key == 'q':
        global loop
        loop = False
    if event.key == ' ':
        plt.close()

count = 0

#neural network setup
X = tf.placeholder(tf.float32, shape=(None, 784), name='X')
labels = tf.placeholder(tf.float32, shape=(None, lettercount), name='Labels')

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
w3 = tf.Variable(name='w3', initial_value=tf.random_normal(shape=(32, lettercount), dtype=tf.float32))
b3 = tf.Variable(name='b3', initial_value=tf.zeros(shape=(1,lettercount), dtype=tf.float32))
y = tf.matmul(z2, w3) + b3

#training setup and parameters
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=y)
step = tf.train.AdamOptimizer().minimize(loss)
BATCH_SIZE = 128
EPOCHS = 80
m = len(trainImages)

#creates training session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training followed by testing every 5 epochs
for epoch in range(EPOCHS):
    for i in range(BATCH_SIZE, m, BATCH_SIZE):
        trainBatch = trainImages[i-BATCH_SIZE:i]
        labelBatch = trainLabels[i-BATCH_SIZE:i]
        sess.run(step, feed_dict={X:trainBatch, labels:labelBatch})
    if (epoch+1) % 5 == 0:
        predictions, test_loss = sess.run([y, loss], feed_dict={X:testImages, labels:testLabels})
        accuracy = np.mean(np.argmax(testLabels, axis=1) == np.argmax(predictions, axis=1))
        print('Epoch: %d\tloss: %1.4f\taccuracy: %1.4f' % (epoch+1, test_loss, accuracy))

#allows for saving and loading models. either run restore or save
#saver = tf.train.Saver()
#saver.restore(sess, "mnist_nn.txt")
#saver.save(sess, "mnist_nn.txt")

#runs the neural network on the images
predictions = sess.run(y, feed_dict={X:testImages})

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
    '''
    name = "Image #{}. Press space to move on, or Q to quit.".format(mplcount-count)

    plt.figure(figsize=[8.8,4]).canvas.set_window_title(name)
    plt.subplot(1, 2, 1)
    plt.imshow(testImages[mplcount].reshape(28, 28).transpose(), cmap='gray')

    x = []
    for i in range(0, lettercount):
        x.append(i)
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
