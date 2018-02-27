import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
def euclidian(x, y):
    return sqrt(np.sum((x-y)**2))

def centroid(cluster):
    return np.sum(cluster, axis=0) / len(cluster)

def kmeans(num_of_clusters, data, num_of_iterations):
    clusters = [[] for _ in range(num_of_clusters)]
    centers = data[:num_of_clusters]
    cut_off = 0.02

    for _ in range(num_of_iterations):
        clusters = [[] for _ in range(num_of_clusters)]
        for i in data:
            min_dst = euclidian(i, centers[0])
            cl_n = 0    
            for j in range(1, num_of_clusters):
                new_dst = euclidian(i, centers[j])
                if new_dst < min_dst:
                    cl_n = j
                    min_dst = new_dst
            clusters[cl_n].append(i)
        
        new_centers = np.array(data[:num_of_clusters])
        
        for i in range(num_of_clusters):
            new_centers[i] = centroid(clusters[i])

        if np.max(np.abs(new_centers - centers)) <= cut_off:
            break
        else:
            centers = new_centers
        
        print([len(c) for c in clusters])
        #plot(data, clusters)
    
    return clusters

def plot(data, clusters):
    matrix = np.zeros((len(data), 3))
    j=0
    for i in range(len(clusters)):
        for c in clusters[i]:
            matrix[j][0]=c[0]
            matrix[j][1]=c[1]
            matrix[j][2]=i
            j+=1
    print(matrix)      

    plt.scatter(matrix[:,0], matrix[:,1], c=matrix[:,2], cmap='rainbow')
    plt.show()

def main():
    data = np.genfromtxt('data.csv', delimiter=',')
    data = np.delete(data[1:], [0], axis=1)
    num_of_clusters = 2
    num_of_iterations = 100
    clusters = kmeans(num_of_clusters, data, num_of_iterations)
    plot(data, clusters)
    
if __name__ == '__main__':
    main()

"""
[1054, 2946]
[800, 3200]
[[ 179.22   95.      0.  ]
 [ 192.34   69.      0.  ]
 [ 140.25   92.      0.  ]
 ..., 
 [  46.85    7.      1.  ]
 [  37.68    7.      1.  ]
 [  50.56    5.      1.  ]]
"""    