# Logistic Regression Using Gradient Descent

## Class: LogisticRegression

### Parameters:

- learning_rate: int, default: 1

- max_iterations: int, default: 100

### Attributes:

- learning_rate: int

- max_iterations: int

- weights: int

### Methods:

| Method                                  | Description                                    |
| --------------------------------------- | ---------------------------------------------- |
| `fit(X, y, initial_weights=None)`       | Fit the model according to given training data |
| `predict(X)`                            | Predict class labels for samples in X          |
| `score(actual_output, expected_output)` | Get Accuracy Score                             |
| `plot_cost_history()`                   | Plot cost history wrt iteration number         |

### Example:

#### Main Function

```
data = np.genfromtxt('train.csv', delimiter=',')
data = data[1:]
train_len = int(len(data) * 0.70)  
train_data = data[:train_len]
test_data = data[train_len:]
clf = LogisticRegression()
clf.fit(train_data[:, :-1], train_data[:, -1])
y = clf.predict(test_data[:, :-1])
print("Accuracy:", clf.score(y, test_data[:, -1]))
clf.plot_cost_history()
```
#### Output
```Accuracy: 0.789189189189```

![alt text](cost_vs_iterations.png "cost_vs_iterations")