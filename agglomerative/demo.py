import numpy as np


def agglomerative(a, labels, num_of_clusters=1):
    a = np.array(a, dtype='float')
    print("Starting...")
    print(*labels)
    # print(a)
    iteration = 1
    while len(a) > num_of_clusters:
        a[a == 0] = np.nan
        i, j = divmod(np.nanargmin(a), a.shape[1])
        for k, row in enumerate(a[i]):
            l = min(a[i][k], a[j][k])
            a[i][k] = a[k][i] = l

        labels[i] = '(' + labels[i].replace(')', '').replace('(', '') + ', ' + \
            labels.pop(j).replace(')', '').replace('(', '') + ')'
        a = np.delete(a, j, axis=0)
        a = np.delete(a, j, axis=1)
        a[np.isnan(a)] = 0
        print("iteration", (iteration), "->", *labels)
        iteration += 1
        # print(a)


labels = ['E', 'A', 'C', 'B', 'D']
a = [[0, 1, 2, 2, 3],
     [1, 0, 2, 5, 3],
     [2, 2, 0, 1, 6],
     [2, 5, 1, 0, 3],
     [3, 3, 6, 3, 0]]

agglomerative(a, labels)

USACities = ['BOS', 'NY', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
USADistances = [
    [0,  206,  429, 1504,  963, 2976, 3095, 2979, 1949],
    [206,    0,  233, 1308,  802, 2815, 2934, 2786, 1771],
    [429,  233,    0, 1075,  671, 2684, 2799, 2631, 1616],
    [1504, 1308, 1075,    0, 1329, 3273, 3053, 2687, 2037],
    [963,  802,  671, 1329,    0, 2013, 2142, 2054,  996],
    [2976, 2815, 2684, 3273, 2013,    0,  808, 1131, 1307],
    [3095, 2934, 2799, 3053, 2142,  808,    0,  379, 1235],
    [2979, 2786, 2631, 2687, 2054, 1131,  379,    0, 1059],
    [1949, 1771, 1616, 2037,  996, 1307, 1235, 1059,    0]]

agglomerative(USADistances, USACities)

"""
Starting...
E A C B D
iteration 1 -> (E, A) C B D
iteration 2 -> (E, A) (C, B) D
iteration 3 -> (E, A, C, B) D
iteration 4 -> (E, A, C, B, D)
Starting...
BOS NY DC MIA CHI SEA SF LA DEN
iteration 1 -> (BOS, NY) DC MIA CHI SEA SF LA DEN
iteration 2 -> (BOS, NY, DC) MIA CHI SEA SF LA DEN
iteration 3 -> (BOS, NY, DC) MIA CHI SEA (SF, LA) DEN
iteration 4 -> (BOS, NY, DC, CHI) MIA SEA (SF, LA) DEN
iteration 5 -> (BOS, NY, DC, CHI) MIA (SEA, SF, LA) DEN
iteration 6 -> (BOS, NY, DC, CHI, DEN) MIA (SEA, SF, LA)
iteration 7 -> (BOS, NY, DC, CHI, DEN, SEA, SF, LA) MIA
iteration 8 -> (BOS, NY, DC, CHI, DEN, SEA, SF, LA, MIA)
"""
