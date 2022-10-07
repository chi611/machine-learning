from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=[[165,45],[150,60],[148,52],[161,58],[178,85],[180,90],[175,84],[192,88]]
train_label=[1,1,1,1,2,2,2,2]


test_data=[[159,48],[167,61],[182,78]]
knn = KNeighborsClassifier()
knn.fit(train_data,train_label)



ans=knn.predict(test_data)
print(ans)
train_data=np.array(train_data)
train_data=train_data.T

test_data=np.array(test_data)
test_data=test_data.T

plt.scatter(train_data[1,0:4],train_data[0,0:4], marker='o',c='#0000FF')
plt.scatter(train_data[1,4:8],train_data[0,4:8], marker='o',c='#FF0000')
#plt.scatter(test_data[1],test_data[0], marker='o',c='#00FF00')

for i in range(len(ans)):
    if ans[i]==1:plt.scatter(test_data[1,i],test_data[0,i], marker='o',c='#CCCCFF')
    if ans[i]==2:plt.scatter(test_data[1,i],test_data[0,i], marker='o',c='#FFCCCC')
plt.axis([0, 100, 0, 200])
plt.savefig('data.png')
plt.show()