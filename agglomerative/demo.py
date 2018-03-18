import numpy as np 
labels = ['E','A', 'C', 'B', 'D']
a = [[0,1,2,2,3],
     [1,0,2,5,3],
     [2,2,0,1,6],
     [2,5,1,0,3],
     [3,3,6,3,0]]

a = np.array(a, dtype='float')

num_of_clusters = 1
print(*labels, sep='    ')
print(a)
while len(a) > num_of_clusters:
    a[a==0]=np.nan
    i,j = divmod(np.nanargmin(a),a.shape[1])
    for k, row in enumerate(a[i]):
        l = min(a[i][k],a[j][k])
        a[i][k]=a[k][i]=l
    labels[i]= labels[i] + '' + labels.pop(j)
    a = np.delete(a, j, axis=0)
    a = np.delete(a, j, axis=1)
    a[np.isnan(a)]=0
    print(*labels, sep='    ')
    print(a)