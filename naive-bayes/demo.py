import pandas as pd
import numpy as np

data = pd.read_csv('./naive-bayes/smsdata.csv', sep='\t')

print(data.head())


train_positive = {}
train_negative = {}
total = len(data)
num_of_spam = len(data[data.label == 'spam'])
pSpam = num_of_spam / total
pHam = (total - num_of_spam) / total
print(pSpam, pHam)


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

test = 'hey sign up today'

pprob = 1
for word in test:
    pprob *= train_positive.get(word, 1) / (total_positive_words)

nprob = 1
for word in test:
    nprob *= train_negative.get(word, 1) / (total_negative_words)

print(pprob*pHam, nprob*pSpam, nprob*pSpam > pprob*pHam)
