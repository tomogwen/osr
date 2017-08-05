# goal: output into a file a 2d numpy array, with the first element containing a second two by two array
# of size [number of examples, 784], containing 78x78 images encoded with 0 or ff as intensity and the
# second element contains a 1d numpy array [number of examples] containing the label for the graph

# id | function
# --------------------
#  0 | pos. linear
#  1 | neg. linear
#  2 | pos. quadratic
#  3 | neg. quadratic
#  4 | sin
#  5 | cos
#  6 | ln
#  7 | exp

import numpy as np
import math


def linear(x, m, c):
    return m * x + c


def quadratic(x, a, b, c):
    return a*x*x + b*x + c


def sin(x, a, b):
    return 3*a*math.sin(b*((math.pi * x)/14))


def cos(x, a, b):
    return 3*a*math.cos(b*((math.pi * x)/14))


def generateCorpus():
    amount = 500
    data = np.ones((amount,784), dtype=np.int)
    labels = np.zeros((amount, 6))
    count = 0

    for i in range(1, 4):
        for j in range(0, 8):
            mapArray = np.ones((28, 28), dtype=np.int)
            for x in range(-13,15):
                y = math.floor(linear(x, i, j-4))
                if (-13 <= x <= 14) and (-13 <= y <= 14):
                    mapArray[14-y][x+13] = 0
            mapFlat = np.ndarray.flatten(mapArray)
            data[count] = mapFlat
            labels[count] = np.array([1,0,0,0,0,0])
            count += 1

    for i in range(1, 4):
        for j in range(0, 8):
            mapArray = np.ones((28, 28), dtype=np.int)
            for x in range(-13,15):
                y = math.floor(linear(x, -i, j-4))
                if (-13 <= x <= 14) and (-13 <= y <= 14):
                    mapArray[14-y][x+13] = 0

            mapFlat = np.ndarray.flatten(mapArray)
            data[count] = mapFlat
            labels[count] = np.array([0,1,0,0,0,0])
            count += 1


    for i in range(1, 2):
        for j in range(1, 3):
            for k in range(0, 8):
                mapArray = np.ones((28, 28), dtype=np.int)
                for x in range(-13,15):
                    y = math.floor(quadratic(x, i, j, k-4))
                    if (-13 <= x <= 14) and (-13 <= y <= 14):
                        mapArray[14-y][x+13] = 0
                mapFlat = np.ndarray.flatten(mapArray)
                data[count] = mapFlat
                labels[count] = np.array([0,0,1,0,0,0])
                count += 1


    for i in range(1, 2):
        for j in range(1, 3):
            for k in range(0, 8):
                mapArray = np.ones((28, 28), dtype=np.int)
                for x in range(-13,15):
                    y = math.floor(quadratic(x, -i, j, k))
                    if (-13 <= x <= 14) and (-13 <= y <= 14):
                        mapArray[14-y][x+13] = 0

                mapFlat = np.ndarray.flatten(mapArray)
                data[count] = mapFlat
                labels[count] = np.array([0,0,0,1,0,0])
                count += 1


    for i in range(1, 3):
        for j in range(1, 5):
            mapArray = np.ones((28, 28), dtype=np.int)
            for x in range(-13,15):
                y = math.floor(sin(x, i, j))
                if (-13 <= x <= 14) and (-13 <= y <= 14):
                    mapArray[14-y][x+13] = 0
            mapFlat = np.ndarray.flatten(mapArray)
            data[count] = mapFlat
            labels[count] = np.array([0,0,0,0,1,0])
            count += 1

    for i in range(1, 3):
        for j in range(1, 5):
            mapArray = np.ones((28, 28), dtype=np.int)
            for x in range(-13,15):
                y = math.floor(cos(x, i, j))
                if (-13 <= x <= 14) and (-13 <= y <= 14):
                    mapArray[14-y][x+13] = 0

            mapFlat = np.ndarray.flatten(mapArray)
            data[count] = mapFlat
            labels[count] = np.array([0,0,0,0,0,1])
            count += 1

    corpus = [data, labels]
    return [corpus, count]

"""corpus = generateCorpus()
print corpus[0][0][21] # """