import pandas as pd
import random

data = pd.read_csv('smsdata.csv', sep='\t')

print(data.head(), end='\n\n')

train_positive = {}
train_negative = {}
total = len(data)
num_of_spam = len(data[data.label == 'spam'])
pSpam = num_of_spam / total
pHam = (total - num_of_spam) / total

print("Probability of Spam:", pSpam)
print("Probability of Ham", pHam)

total_negative_words = 0
total_positive_words = 0
for i, email in data.iterrows():
    for word in email.body:
        if email.label == 'spam':
            train_negative[word] = train_negative.get(word, 0) + 1
            total_negative_words += 1
        else:
            train_positive[word] = train_positive.get(word, 0) + 1
            total_positive_words += 1


print(total_negative_words, total_positive_words)


def classify(test):
    pprob = 1.0
    for word in test:
        pprob *= train_positive.get(word, 0) / (total_positive_words)

    nprob = 1.0
    for word in test:
        nprob *= train_negative.get(word, 0) / (total_negative_words)

    return 'spam' if nprob*pSpam > pprob*pHam else 'ham'


test = data.sample(frac=0.1)

correct = 0
for i, t in test.iterrows():
    label = classify(t)
    print(label, t.label)
    if t.label == label:
        correct += 1
print("Accuracy: ", correct / len(test))
