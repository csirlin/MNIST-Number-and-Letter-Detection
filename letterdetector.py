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

print("Running MNIST")

trI = mnist.train.images
trL = mnist.train.labels
teI = mnist.test.images
teL = mnist.test.labels

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


combined = list(zip(trainImages, trainLabels))
np.random.shuffle(combined)
trainImages[:], trainLabels[:] = zip(*combined)

combined = list(zip(testImages, testLabels))
np.random.shuffle(combined)
testImages[:], testLabels[:] = zip(*combined)

trainLabelsTemp = np.zeros((124800, 26))
testLabelsTemp = np.zeros((20800, 26))

print(len(trainLabels))
print(len(testLabels))

for i in range(0, len(trainLabels)):
    value = int(trainLabels[i])-1
    trainLabelsTemp[i][value] = 1

for i in range(0, len(testLabels)):
    value = int(testLabels[i])-1
    testLabelsTemp[i][value] = 1

trainLabels = np.copy(trainLabelsTemp)
testLabels = np.copy(testLabelsTemp)

print("Not working MNIST pictures: " + str(trainImages[0]))
print("Not working MNIST labels: " + str(trainLabels[0]))

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

print("Not working MNIST pictures: " + str(trainImages[0]))
print("Not working MNIST labels: " + str(trainLabels[0]))

'''
#mnist data for comparison
mtrainImages = mnist.train.images
mtrainLabels = mnist.train.labels
mtestImages = mnist.test.images
mtestLabels = mnist.test.labels

print(mtestLabels.shape)
print(mtestLabels[1])

#emnist
trainLabels = np.copy(idx2numpy.convert_from_file('emnist-letters-train-labels-idx1-ubyte.txt'))
trainImages = np.copy(idx2numpy.convert_from_file('emnist-letters-train-images-idx3-ubyte.txt'))
testLabels = np.copy(idx2numpy.convert_from_file('emnist-letters-test-labels-idx1-ubyte.txt'))
testImages = np.copy(idx2numpy.convert_from_file('emnist-letters-test-images-idx3-ubyte.txt'))

print(mtrainImages.flags)
print(trainImages.flags)

#shuffling the datasets
combined = list(zip(mtrainImages, mtrainLabels))
np.random.shuffle(combined)
mtrainImages[:], mtrainLabels[:] = zip(*combined)

combined = list(zip(mtestImages, mtestLabels))
np.random.shuffle(combined)
mtestImages[:], mtestLabels[:] = zip(*combined)


combined = list(zip(trainImages, trainLabels))
np.random.shuffle(combined)
trainImages[:], trainLabels[:] = zip(*combined)

combined = list(zip(testImages, testLabels))
np.random.shuffle(combined)
testImages[:], testLabels[:] = zip(*combined)

#testing with 10 letters
trl10 = []
tri10 = []
tel10 = []
tei10 = []

for i in range(0, len(trainLabels)):
    if trainLabels[i] <= 10:
        trl10.append(trainLabels[i])
        tri10.append(trainImages[i])

for i in range(0, len(testLabels)):
    if testLabels[i] <= 10:
        tel10.append(testLabels[i])
        tei10.append(testImages[i])

trainLabels = np.copy(trl10)
trainImages = np.copy(tri10)
testLabels = np.copy(tel10)
testImages = np.copy(tei10)


#images are stored column by column for some reason, this fixes that.
for i in range(0, len(trainImages)):
    trainImages[i] = trainImages[i].transpose()
for i in range(0, len(testImages)):
    testImages[i] = testImages[i].transpose()
'''
'''
#examples from both training and testing datasets
for i in range(0, 10):
    print("Training set examples: {} ({})".format(chr(trainLabels[i]+64), trainLabels[i]))
    plt.imshow(np.asarray(trainImages[i]).reshape(28,28), cmap='gray')
    plt.show()
for i in range(0, 10):
    print("Test set examples: {} ({})".format(chr(testLabels[i]+64), testLabels[i]))
    plt.imshow(np.asarray(testImages[i]).reshape(28,28), cmap='gray')
    plt.show()
'''

#matplotlib event listener
loop = True
def press(event):
    if event.key == 'q':
        global loop
        loop = False
    if event.key == ' ':
        plt.close()
print(trainLabels.shape)
print(trainImages.shape)

'''
trainLabelsTemp = np.zeros((48000, 10))
testLabelsTemp = np.zeros((8000, 10))

#changing the format
for i in range(0, len(trainLabels)):
    value = int(trainLabels[i])-1
    trainLabelsTemp[i][value] = 1

for i in range(0, len(testLabels)):
    value = int(testLabels[i])-1
    testLabelsTemp[i][value] = 1

trainLabels = trainLabelsTemp
testLabels = testLabelsTemp
trainImages = trainImages.reshape(48000, 784)
testImages = testImages.reshape(8000, 784)

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

print("MNIST" + str(mtestImages.shape) + "EMNIST" + str(testImages.shape))
print("MNIST" + str(mtrainImages.shape) + "EMNIST" + str(trainImages.shape))
print("MNIST" + str(mtestLabels.shape) + "EMNIST" + str(testLabels.shape))
print("MNIST" + str(mtrainLabels.shape) + "EMNIST" + str(trainLabels.shape))
print("MNSIT" + str(mtestLabels[0]) + "EMNIST" + str(testLabels[1]))
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
BATCH_SIZE = 64
EPOCHS = 10
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
    #if (epoch+1) % 5 = 0:
    predictions, test_loss = sess.run([y, loss], feed_dict={X:testImages, labels:testLabels})
    accuracy = np.mean(np.argmax(testLabels, axis=1) == np.argmax(predictions, axis=1))
    print('Epoch: %d\tloss: %1.4f\taccuracy: %1.4f' % (epoch+1, test_loss, accuracy))

#allows for saving and loading models. either run restore or save
saver = tf.train.Saver()
#saver.restore(sess, "mnist_nn.txt")
#saver.save(sess, "lmnist_nn.txt")

#runs the neural network on the images
predictions = sess.run(y, feed_dict={X:testImages})
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
