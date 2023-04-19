import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

main_path = "results/test1"
cmp1 = "fed"
cmp2 = "non_fed"
tar = "f1_test.txt"

data1 = pd.read_csv(os.path.join(main_path, cmp1, tar), header=None)
data2 = pd.read_csv(os.path.join(main_path, cmp2, tar), header=None)


plotSize = (6 * 1.618, 6)

fig, ax = plt.subplots(figsize=plotSize)

ax.plot(data1.iloc[:, 3:-1:2])
ax.plot(data2.iloc[:, 3:-1:2])

print(
    data1.iloc[:, 3:-1:2].max(axis=0),
)
print(data2.iloc[:, 3:-1:2].max(axis=0))

# ax.plot(data1.iloc[:, 3:])

# ax.plot(data2.iloc[:, 3:])


plt.savefig(os.path.join(main_path, "test.png"))
