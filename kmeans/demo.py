import numpy as np
import matplotlib.pyplot as plt

def euclidian(x, y):
    return np.sqrt(np.sum((x-y)**2))

def centroid(cluster):
    if len(cluster) <= 0:
        return 0
    return np.sum(cluster, axis=0) / len(cluster)

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
    # centers = np.array([ data[:,:-1].mean(axis=0) for _ in range(num_of_clusters)])
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

def plot(matrix):
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
    train_len = int(len(data) * 0.90)
    train_data = data[:train_len]
    test_data = data[train_len:]
    #train_data = data
    clusters, centers = kmeans(num_of_clusters, train_data, num_of_iterations)
    print("final clusters")    
    print(clusters)
    centers = expand(centers)
    centers[:,-1] += num_of_clusters
    plot(np.append(clusters, centers, axis=0))

    predicted = classify(test_data, centers)
    print("predicted clusters")
    print(predicted)
    
if __name__ == '__main__':
    main()

"""
initial clusters
[[  71.24   28.      0.  ]
 [  52.53   25.      0.  ]
 [  64.54   27.      0.  ]
 ..., 
 [ 161.76    5.      0.  ]
 [ 169.54   12.      0.  ]
 [ 173.72   10.      0.  ]]
iter => 1 cluster count [95, 305, 2891, 309]
iter => 2 cluster count [101, 299, 2804, 396]
iter => 3 cluster count [103, 297, 2785, 415]
iter => 4 cluster count [103, 297, 2777, 423]
iter => 5 cluster count [103, 297, 2775, 425]
iter => 6 cluster count [103, 297, 2775, 425]
final clusters
[[  71.24   28.      3.  ]
 [  52.53   25.      3.  ]
 [  64.54   27.      3.  ]
 ..., 
 [ 161.76    5.      1.  ]
 [ 169.54   12.      1.  ]
 [ 173.72   10.      1.  ]]
predicted clusters
[[ 180.98    8.      1.  ]
 [ 181.89   17.      1.  ]
 [ 172.85   13.      1.  ]
 ..., 
 [ 170.91   12.      1.  ]
 [ 176.14    5.      1.  ]
 [ 168.03    9.      1.  ]]
"""    