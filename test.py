import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

main_path = "NEW_DATA/emmanuel12"
cmp1 = "fed"
cmp2 = "non"

tar = "loss_train.txt"
data1 = pd.read_csv(os.path.join(main_path, cmp1, tar), header=None)
data2 = pd.read_csv(os.path.join(main_path, cmp2, tar), header=None)


plotSize = (6 * 1.618, 6)

fig, ax = plt.subplots(figsize=plotSize)
dev_size = len(data1.iloc[:, 3:].columns)
labels = ["dev"+str(i) for i in range(dev_size)]
ax.plot(data1.iloc[:, 3:], label = labels)

ax.plot(data2.iloc[:, 3:], "--",label = labels)

plt.grid(True)
plt.legend()
plt.savefig(os.path.join(main_path, "loss_train.png"))

plt.close()


tar = "f1_test.txt"
data1 = pd.read_csv(os.path.join(main_path, cmp1, tar), header=None)
data2 = pd.read_csv(os.path.join(main_path, cmp2, tar), header=None)


plotSize = (6 * 1.618, 6)

fig, ax = plt.subplots(figsize=plotSize)

ax.plot(data1.iloc[:, 3:-1:2], label = labels)
ax.plot(data2.iloc[:, 3:-1:2],  "--", label = labels)

print("FED:")
print(data1.iloc[:, 3:-1:2].max(axis=0))

print("NO_FED:")
print(data2.iloc[:, 3:-1:2].max(axis=0))
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(main_path, "f1.png"))
