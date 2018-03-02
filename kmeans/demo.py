import numpy as np
import matplotlib.pyplot as plt
import random

def euclidian(x, y):
    return np.sqrt(np.sum((x-y)**2))

def centroid(cluster):
    if len(cluster) == 0:
        return 0
    return np.mean(cluster, axis=0)

def get_centers(num_of_clusters, data):
    centers = np.zeros([num_of_clusters, len(data[0])-1])
    for i in range(num_of_clusters):
        points = data[data[:, -1] == i]
        points = points[:,:-1]
        centers[i] = centroid(points)
    return centers

def expand(data):
    return np.append(data, np.zeros([len(data), 1]), axis=1)  

def get_cluster(point, centers):
    distances = np.array([euclidian(point, j) for j in centers])
    return np.argmin(distances)

def kmeans(num_of_clusters, data, num_of_iterations):
    data = expand(data)
    print("initial clusters")
    print(data)
    centers = np.array([[175,80], [175,15], [50, 15], [50,35]])
    #centers = np.array([ data[:,:-1].mean(axis=0) for _ in range(num_of_clusters)])
    #print(centers)

    for itr in range(1, num_of_iterations):
        for i in range(len(data)):
            data[i][-1] = get_cluster(data[i,:-1], centers)

        print("iter =>", itr, "cluster count", points_count(num_of_clusters, data))
        
        new_centers = get_centers(num_of_clusters, data)
        #print(new_centers)
        #print(np.abs(new_centers - centers))
        
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers
        #plot(data, clusters)
    
    return data, centers

def plot(clusters, centers):
    centers = expand(centers)
    centers[:,-1] += len(centers)
    matrix = np.append(clusters, centers, axis=0)
    plt.scatter(matrix[:,0], matrix[:,1], c=matrix[:,2], cmap='rainbow')
    plt.show()

def points_count(num_of_clusters,data):
    return [len(data[data[:,-1] == i]) for i in range(num_of_clusters)]

def classify(data, centers):
    data = expand(data)
    for i in range(len(data)):
        data[i][-1] = get_cluster(data[i,:-1], centers)
    return data
    

def main():
    data = np.genfromtxt('data.csv', delimiter=',')
    data = np.delete(data[1:], [0], axis=1)
    num_of_clusters = 4
    num_of_iterations = 100
    random.shuffle(data)
    train_len = int(len(data) * 0.90)
    train_data = data[:train_len]
    test_data = data[train_len:]
    #train_data = data
    clusters, centers = kmeans(num_of_clusters, train_data, num_of_iterations)
    print("final clusters")    
    print(clusters)
    plot(clusters, centers)

    predicted = classify(test_data, centers)
    print("predicted clusters")
    print(predicted)
    plot(predicted, centers)
    
if __name__ == '__main__':
    main()

"""
initial clusters
[[ 71.24  28.     0.  ]
 [ 71.24  28.     0.  ]
 [ 52.53  25.     0.  ]
 ..., 
 [ 39.02   4.     0.  ]
 [ 69.14  40.     0.  ]
 [ 60.97   9.     0.  ]]
iter => 1 cluster count [14, 8, 2662, 916]
iter => 2 cluster count [13, 9, 2443, 1135]
iter => 3 cluster count [13, 9, 2334, 1244]
iter => 4 cluster count [13, 9, 2302, 1276]
iter => 5 cluster count [13, 9, 2300, 1278]
iter => 6 cluster count [13, 9, 2300, 1278]
final clusters
[[ 71.24  28.     3.  ]
 [ 71.24  28.     3.  ]
 [ 52.53  25.     3.  ]
 ..., 
 [ 39.02   4.     2.  ]
 [ 69.14  40.     3.  ]
 [ 60.97   9.     2.  ]]
predicted clusters
[[  48.16    6.      2.  ]
 [  51.2     6.      2.  ]
 [  60.4     7.      2.  ]
 ..., 
 [ 179.97    0.      1.  ]
 [  32.53    4.      2.  ]
 [  50.2     5.      2.  ]]
"""    