import numpy as np
from collections import Counter, defaultdict
import json


def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob


def naive_bayes(training, outcome, new_sample):
    classes = np.unique(outcome)
    rows, cols = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)

    class_probabilities = occurrences(outcome)

    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset = training[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:, j])

    for cls in classes:
        for j in range(0, cols):
            likelihoods[cls][j] = occurrences(likelihoods[cls][j])

    print(json.dumps(likelihoods, indent=True))

    results = {}
    for cls in classes:
        class_probability = class_probabilities[cls]
        for i in range(0, len(new_sample)):
            relative_values = likelihoods[cls][i]
            if new_sample[i] in relative_values.keys():
                class_probability *= relative_values[new_sample[i]]
            else:
                class_probability *= 0
            results[cls] = class_probability
    print("Prediction for", new_sample)
    print(results)


if __name__ == "__main__":
    data = np.genfromtxt('flu.csv', delimiter=',')
    training = data[1:, :-1]
    outcome = data[1:, -1]
    new_sample = np.asarray((1, 0, 1, 0))
    naive_bayes(training, outcome, new_sample)

"""
{
 "0.0": {
  "0": {
   "1.0": 0.3333333333333333,
   "0.0": 0.6666666666666666
  },
  "1": {
   "0.0": 0.6666666666666666,
   "1.0": 0.3333333333333333
  },
  "2": {
   "1.0": 0.3333333333333333,
   "0.0": 0.3333333333333333,
   "2.0": 0.3333333333333333
  },
  "3": {
   "1.0": 0.3333333333333333,
   "0.0": 0.6666666666666666
  }
 },
 "1.0": {
  "0": {
   "1.0": 0.6,
   "0.0": 0.4
  },
  "1": {
   "1.0": 0.8,
   "0.0": 0.2
  },
  "2": {
   "0.0": 0.2,
   "2.0": 0.4,
   "1.0": 0.4
  },
  "3": {
   "0.0": 0.2,
   "1.0": 0.8
  }
 }
}
Prediction for [1 0 1 0]
{0.0: 0.018518518518518517, 1.0: 0.006000000000000002}
"""
