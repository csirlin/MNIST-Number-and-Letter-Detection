#import statements
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
import copy
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#import mnist data and print basic info about it
trainImages = mnist.train.images
trainLabels = mnist.train.labels
testImages = mnist.test.images
testLabels = mnist.test.labels
print(type(trainImages))
#print('trainImages:', trainImages.shape)
#print('trainLabels:', trainLabels.shape)
#print('testImages:', testImages.shape)
#print('testLabels:', testLabels.shape)

#matplotlib event listener
loop = True
def press(event):
    if event.key == 'q':
        global loop
        loop = False
    if event.key == ' ':
        plt.close()

#setup for creating images
baseimg = mpimg.imread('numberbase.png')
print("Make your own numbers!")
count = 0
blankimg = copy.deepcopy(baseimg)
createdLabels = []

#loop for creating images
while loop == True:

    plt.imshow(blankimg)
    plt.imsave('images/img{}.png'.format(count), blankimg)

    plt.connect("key_press_event", press)
    os.system('open images/img{}.png'.format(count))
    plt.show()

    print("What number is this?")
    number = int(input())

    thing = [0,0,0,0,0,0,0,0,0,0]
    thing[number] = 1
    createdLabels.append(thing)
    count += 1


createdLabels = np.asarray(createdLabels)

#put all the created images in a loop
createdImages=[None]*count
for i in range(0,count):
    img = mpimg.imread('images/img{}.png'.format(i))
    img2=[]
    for j in range(0,28):
        for k in range(0,28):
            img2.append(img[j][k][0])
    createdImages[i] = img2

#shuffle
combined = list(zip(trainImages, trainLabels))
np.random.shuffle(combined)
trainImages[:], trainLabels[:] = zip(*combined)

combined = list(zip(testImages, testLabels))
np.random.shuffle(combined)
testImages[:], testLabels[:] = zip(*combined)



'''
#view some images
for index in range (0,5):
    if loop == True:
        print(np.argmax(trainLabels[index]))
        plt.imshow(trainImages[index].reshape(28, 28), cmap='gray')
        plt.connect("key_press_event", press)
        plt.show()

print(np.sum(trainLabels, axis=0))
loop = True

#show the average drawing of each number
for digit in range(0,10):
    if loop == True:
        plt.imshow(trainImages[np.argmax(trainLabels, axis=1) == digit].mean(axis=0).reshape(28,28), cmap='gray')
        plt.connect("key_press_event", press)
        plt.show()
'''

#neural network setup
X = tf.placeholder(tf.float32, shape=(None, 784), name='X')
labels = tf.placeholder(tf.float32, shape=(None, 10), name='Labels')

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
w3 = tf.Variable(name='w3', initial_value=tf.random_normal(shape=(32, 10), dtype=tf.float32))
b3 = tf.Variable(name='b3', initial_value=tf.zeros(shape=(1,10), dtype=tf.float32))
y = tf.matmul(z2, w3) + b3

#training setup and parameters
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=y)
step = tf.train.AdamOptimizer().minimize(loss)
BATCH_SIZE = 32
EPOCHS = 40
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
saver = tf.train.Saver()
saver.restore(sess, "mnist_nn.txt")
#saver.save(sess, "mnist_nn.txt")

#manually created image
newImage = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

newImage = np.asarray(newImage)

#appends user-created images and their labels to the lists
testImages = np.insert(testImages, 0, newImage, axis=0)
testImages = np.insert(testImages, 0, np.asarray(createdImages), axis=0)
testLabels = np.insert(testLabels, 0, np.asarray([0,0,1,0,0,0,0,0,0,0]), axis=0)
testLabels = np.insert(testLabels, 0, np.asarray(createdLabels), axis=0)

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

    if mplcount < count:
        name = "Drawn Image #{}:".format(mplcount+1)
    elif mplcount == count:
        name = "Raw Data 2:"
    else:
        name = "MNIST Image #{}:".format(mplcount-count)

    plt.figure(figsize=[8.8,4]).canvas.set_window_title(name)
    plt.subplot(1, 2, 1)
    plt.imshow(testImages[mplcount].reshape(28, 28), cmap='gray')
    x = [0,1,2,3,4,5,6,7,8,9]
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
