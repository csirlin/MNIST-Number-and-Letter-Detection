
'''
trL = open("emnist-letters-train-labels-idx1-ubyte.txt", "r").read()
trI = open("emnist-letters-train-images-idx3-ubyte.txt", "r").read()
teL = open("emnist-letters-test-labels-idx1-ubyte.txt", "r").read()
teI = open("emnist-letters-test-images-idx3-ubyte.txt", "r").read()


with open("emnist-letters-train-labels-idx1-ubyte.txt", "rb") as binary_file:
    # Read the whole file at once
    trL = int.from_bytes(binary_file.read(), byteorder='big')
    print(trL)
    trL = str(trL)
    trL = trL.replace("\\r", "").replace("\\n", "").replace("\\t", "").split("\\x")
    print(trL)

with open("emnist-letters-train-labels-idx1-ubyte.txt", "rb") as binary_file:
    # Read the whole file at once
    trL = binary_file.read()
    print(trL)
    trL = str(trL)
    trL = trL.replace("\\r", "").replace("\\n", "").replace("\\t", "").split("\\x")
    print(trL)
'''
'''

for i in range(0, len(trL)):
    temp = np.ndarray(shape=(28,28))
    for j in range(0, 28):
        for k in range(0,28):
            temp[k][j] = trL[i][j][k]
    for j in range(0, 28):
        for k in range(0,28):
            trl[i][j][k] = temp[j][k]
'''

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#import mnist data and print basic info about it
'''
#00 00 08 01 00 01 e7 80 17 07 10 0f 17 11 0d 0b | 00 00 08 03 00 00 27 10 00 00 00 1c 00 00 00 1c
testarray = "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 04 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 09 20 25 27 52 7c 4d 08 00 00 00 00 00 00 00 00 00 00 00 00 00 02 09 20 25 25 27 52 8b cc d7 d9 e9 f9 da 5a 07 00 00 00 00 00 02 04 04 04 05 15 22 52 8b cc d9 d9 d9 e9 fa fe fe fe fe ff fd c8 20 00 00 00 00 07 4c 7d 7f 7f 81 ac cc e9 fa fe fe fe fe fe fe fa fa fa fc fe fc ac 15 00 00 00 00 13 97 d7 d9 d9 d9 e9 f5 fc fe fe fe fc fa fa f5 de d9 d9 ec fe fa8b 09 00 00 00 00 14 a8 f9 fe fe fe fe fc f5 de d9 d7 ac 81 7f 72 33 27 29 92 f9 da 4d 02 00 00 00 00 02 43 aa d7 d9 d9 cc ac 73 33 25 25 15 05 04 04 00 00 09 87 cb 5a 08 00 00 00 00 00 00 02 15 25 25 25 20 15 04 00 00 00 00 00 00 00 00 00 1b 91 3a 07 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 05 1a 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00".split(" ") #0e 08 08 06 08 04 06 17
print(testarray[1])
print(type(testarray[1]))
print(len(testarray))

def hex_to_decimal(hex)
for i in range(0, len(testarray)):
    hexx = testarray[i]
    number = 0
    ones = 0
    sixteens = 0
    if hexx[0] == '0':
        ones = 0
    elif hexx[0] == '1':
        ones = 1
    elif hexx[0] == '2':
        ones = 2
    elif hexx[0] == '3':
        ones = 3
    elif hexx[0] == '4':
        ones = 4
    elif hexx[0] == '5':
        ones = 5
    elif hexx[0] == '6':
        ones = 6
    elif hexx[0] == '7':
        ones = 7
    elif hexx[0] == '8':
        ones = 8
    elif hexx[0] == '9':
        ones = 9
    elif hexx[0] == 'a':
        ones = 10
    elif hexx[0] == 'b':
        ones = 11
    elif hexx[0] == 'c':
        ones = 12
    elif hexx[0] == 'd':
        ones = 13
    elif hexx[0] == 'e':
        ones = 14
    elif hexx[0] == 'f':
        ones = 15

    if hexx[1] == '0':
        sixteens = 0
    elif hexx[1] == '1':
        sixteens = 1
    elif hexx[1] == '2':
        sixteens = 2
    elif hexx[1] == '3':
        sixteens = 3
    elif hexx[1] == '4':
        sixteens = 4
    elif hexx[1] == '5':
        sixteens = 5
    elif hexx[1] == '6':
        sixteens = 6
    elif hexx[1] == '7':
        sixteens = 7
    elif hexx[1] == '8':
        sixteens = 8
    elif hexx[1] == '9':
        sixteens = 9
    elif hexx[1] == 'a':
        sixteens = 10
    elif hexx[1] == 'b':
        sixteens = 11
    elif hexx[1] == 'c':
        sixteens = 12
    elif hexx[1] == 'd':
        sixteens = 13
    elif hexx[1] == 'e':
        sixteens = 14
    elif hexx[1] == 'f':
        sixteens = 15

    number = (16*sixteens + ones)/255
    testarray[i] = round(number, 2)

plt.imshow(np.asarray(testarray).reshape(28,28), cmap='gray')
plt.show()

for features in mnist.take(1):
  image, label = features["image"], features["label"]


trainImages = mnist.train.images
trainLabels = mnist.train.labels
testImages = mnist.test.images
testLabels = mnist.test.labels

#print('trainImages:', trainImages.shape)
#print('trainLabels:', trainLabels.shape)
#print('testImages:', testImages.shape)
#print('testLabels:', testLabels.shape)
