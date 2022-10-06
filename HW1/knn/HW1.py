from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=[[165,45],[150,60],[148,52],[161,58],[178,85],[180,90],[175,84],[192,88]]
train_label=[1,1,1,1,2,2,2,2]


test_data=[[159,48],[167,61],[182,78]]
knn = KNeighborsClassifier()
knn.fit(train_data,train_label)

print(knn.predict(test_data))
print(test_data)

train_data=np.array(train_data)
train_data=train_data.T

test_data=np.array(test_data)
test_data=test_data.T

plt.scatter(train_data[1,0:4],train_data[0,0:4], marker='o',c='b')
plt.scatter(train_data[1,4:8],train_data[0,4:8], marker='o',c='r')
plt.scatter(test_data[1],test_data[0], marker='o',c='g')

plt.axis([0, 100, 0, 200])
#plt.savefig('data.png')
plt.show()