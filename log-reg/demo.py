import numpy as np
from math import log, exp

def z(theta, data): 
    return np.dot(theta, np.append(1,data))    

def hox(z):
    #print(z)
    return 1.0 / (1 + exp(-z))

def cost(hx, y):
    #print('hx=', hx, 'y=', y)
    return -y * log(hx) - (1-y) * log(1-hx)

def j_theta(theta, data):
    total = 0
    for i in data:
        total += cost(hox(z(theta, i[:-1])), i[-1])
    return total / len(data)

def single_step(theta, data, learning_rate):
    new_theta = np.zeros(len(theta))
    for j in range(len(theta)):
        total = 0
        for i in range(len(data)):
            if j == 0:
                total += (hox(z(theta, data[i][:-1])) - data[i][-1])
            else:
                total += (hox(z(theta, data[i][:-1])) - data[i][-1]) * data[i][j-1] 
            
        new_theta[j] = theta[j] - learning_rate * (total / len(data))
    return new_theta

def classify(input, weights):
    p = hox(z(weights, input))
    return 1 if p >= 0.5 else 0 

def main():
    learning_rate = 1
    num_iterations = 100
    data = np.genfromtxt('train.csv', delimiter=',')
    data = data[1:]
    train_len = int(len(data) * 0.90)
    train_data = data[:train_len]
    test_data = data[train_len:]
    theta = np.zeros(len(train_data[0]))
    jtheta = j_theta(theta, train_data)
    print("Before training, J(@): ", jtheta)

    for i in range(num_iterations):
        theta = single_step(theta, train_data, learning_rate)
        #print(theta)
        newjt = j_theta(theta, train_data)
        i+=1
        if newjt < jtheta:
            jtheta = newjt
            #print(i, '->', jtheta)
        else:
            break
        
    jtheta = j_theta(theta, train_data)
    print(i, "After training, J(@): ", jtheta)
    
    correct_predictions = 0
    for i in test_data:
        predicted_output = classify(i[:-1], theta)
        print("prediction->", predicted_output, " actual->", i[-1])
        if predicted_output == i[-1]:
            correct_predictions += 1
    
    print("Accuracy is", correct_predictions / len(test_data))


if __name__ == '__main__':
    main()
"""
Before training, J(@):  2.157682840216067
After training, J(@):  0.5210831948994271
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
prediction-> 0  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 0.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 0.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 1  actual-> 1.0
prediction-> 0  actual-> 0.0
Accuracy is 0.8064516129032258
"""