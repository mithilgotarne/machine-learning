from numpy import *
from sklearn.linear_model import LinearRegression

def score(b, m, points):
    y_mean = mean(points[:,1])
    s_res, s_tot = 0, 0
    for x, y in points:
        s_res += (y - (m*x + b)) **2
        s_tot += (y - y_mean) **2    
        
    return 1 - (s_res / s_tot)
    
def compute_error(b, m, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        total_error += (y -(m*x + b)) ** 2
    return total_error / len(points)
    
def step_gradient(b, m, points, learning_rate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = len(points)
    for i in range(N):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2/N) * (y - (m*x + b))
        m_gradient += -(2/N) * x * (y - (m*x + b)) 
    
    return b - (learning_rate * b_gradient) , m - (learning_rate * m_gradient)
         
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    error = compute_error(b, m, points)

    print(b, m, error)
    
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
        new_error = compute_error(b,m,points)
        if new_error <= error:
            error = new_error
            #print(b, m, compute_error(b,m, points), score(b,m, points))
            
        else:
            print(i, new_error)
            break
    
    return b, m
    
def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0004
    #y=mx+c
    initial_m = 0
    initial_b = 0
    num_interations = 200000
    b, m = gradient_descent_runner(points, initial_b, initial_m,learning_rate, num_interations)
    print(b, m, compute_error(b,m, points), score(b,m, points))

    lr = LinearRegression()
    lr.fit(points[:,0].reshape(-1,1), points[:,1])
    sc = lr.score(points[:,0].reshape(-1,1), points[:,1])    
    print(lr.intercept_ ,lr.coef_, sc)

if __name__ == '__main__':
    run()