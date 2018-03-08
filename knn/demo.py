import numpy as np
import matplotlib.pyplot as plt
import random

def euclidian(x,y):
    return np.sqrt(np.sum((x-y)**2))

def get_class(distances):
    count = {}
    for i in distances:
        point_class = i[1][-1]
        if point_class not in count.keys():
            count[point_class] = 1
        else:
            count[point_class] += 1
    return sorted(count, key=count.get)[-1]

if __name__ == '__main__':
    num_of_points = 5
    data = np.genfromtxt('data.csv', delimiter=',')

    data = data [1:,1:]

    random.shuffle(data)

    train_data_len = int(len(data)*0.60)
    train_data = data[:train_data_len]
    test_data = data[train_data_len:]

    correct = 0

    for point in test_data:
        distances = [(euclidian(point[:-1],p[:-1]), p) for p in train_data]
        distances = sorted(distances,key= lambda x: x[0])[:5]
        point_class = get_class(distances)
        print("Actual->", point[-1], " Predicted->", point_class)
        if point_class == point[-1]:
            correct += 1
    print("Accuracy:", correct/len(test_data))
"""
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 4.0  Predicted-> 4.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Actual-> 2.0  Predicted-> 2.0
Accuracy: 0.9535714285714286
"""