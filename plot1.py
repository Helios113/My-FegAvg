import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
from matplotlib.ticker import MaxNLocator
import matplotlib
cmap = matplotlib.colormaps['tab20']


fed = False
cmp1 = "fed"
cmp2 = "non"
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}
matplotlib.rc('font', **font)

x = [-2,-1,0,1,2,3,4]
y = [[0.820211,0.576096, 0.848620,0.375000,0.553571, 0.443385,0.644481,0.754261,0.650974,0.540787,0.282670,0.267045,0.217330,0.439935,0.316761],[0.377841,0.724838,0.604708,0.796063,0.861404,0.772524],[0.899148],[0.882102],[0.867898],[0.855114],[0.846591]]
plotSize = (6 * 1.618, 6)
fig, ax = plt.subplots(figsize=plotSize)
for i,j in enumerate(y):
	x1 = [x[i]]*len(j)
	print(x1)
	print(j)
	ax.scatter(x1,j, color=cmap(0), )


ax.set_ylim(-0.1,1)
ax.set_xlim(-4,4)
ax.set_ylabel("F1 Score")
ax.set_xlabel(r"$\Delta$ Modalities")

fig.legend(loc=4, prop={'size': 16})
plt.tight_layout()

plt.savefig("test_plot.pdf")
plt.close()

