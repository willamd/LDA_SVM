from sklearn.externals import joblib
import matplotlib.pyplot as plt
data=joblib.load("/home/william/ServerNet/data/rbf20/train_acctop5rbf_2000_275_60.dat")

a=(20,data[-1][1])
data.insert(20,a)
print(data)

plt.plot(*zip(*data[:20]))
max_item=0
for i in range(len(data)):
    if data[i][1]>max_item:
        max_item=data[i][1]
print(max_item)
plt.show()