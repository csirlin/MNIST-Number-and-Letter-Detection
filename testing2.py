#import statements
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.image as mpimg

array = np.full((4,4),200.0)
print(array)

for i in range(0, len(array)):
    for j in range(0, len(array[0])):
        value = array[i][j]
        print(value)

        newvalue = value/256
        print(newvalue)

        array[i][j] = newvalue

print(array)

'''
print(tfds.list_builders())
emnist = tfds.load(name='emnist', split='train')



mnist = input_data.read_data_sets('emnist/letters', one_hot=True)

trainImages = mnist.train.images
trainLabels = mnist.train.labels
testImages = mnist.test.images
testLabels = mnist.test.labels

combined = list(zip(trainImages, trainLabels))
np.random.shuffle(combined)
trainImages[:], trainLabels[:] = zip(*combined)

combined = list(zip(testImages, testLabels))
np.random.shuffle(combined)
testImages[:], testLabels[:] = zip(*combined)


for i in range(0,100):
    plt.imshow(testImages[i].reshape(28,28), cmap='gray')
    plt.show()
'''
'''
#import mnist data and print basic info about it
trainImages = mnist.train.images
trainLabels = mnist.train.labels
testImages = mnist.test.images
testLabels = mnist.test.labels

a = [1,2,3]
b = [4,5]

a = np.asarray(a)
b = np.asarray(b)
print(a)
print(b)

print(np.insert(a, 3, b, axis=0))

for i in range(9,-1,-1):
    print(i)

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(1, 2, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(1, 2, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()

fig, ax = plt.subplots()
ax.plot(np.random.rand(10))

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)

for i in range(0,10):
    print(len(trainImages[i]))
    plt.imshow(trainImages[i].reshape(28,28), cmap = "gray")
    plt.connect("button_press_event", onclick)
    plt.show()
    plt.Rectangle((0,0), 50, 20, fc='blue',ec="red")
    plt.show()
    rawData = trainImages[i]
    picture = [[0] * 28 for i in range(28)]

    for j in range(0, 28):
        for k in range(0, 28):
            print("j:{} k:{} rawdata:{} 28j+k:{}".format(j, k, rawData[28*j+k], 28*j+k))
            #picture[j][k] = [rawData[28*j+k], rawData[28*j+k], rawData[28*j+k]]
            picture[j].append([rawData[28*j+k], rawData[28*j+k], rawData[28*j+k]])
    plt.imsave('images/testing.png',picture)

box = np.arange(40).reshape(4,10)
print(box)
labels = []
for i in range(0,3):
    print("What number is this?")
    number = int(input())

    thing = [0,0,0,0,0,0,0,0,0,0]

    print(thing)
    print(type(thing[0]))
    thing[number] = 1
    print(thing)
    labels.append(thing)

print(labels)
labels2 = np.asarray(labels)
print(labels2)

box = np.insert(box, 0, np.asarray(labels), axis=0)
print(box)
'''
