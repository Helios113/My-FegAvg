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

x = [12,8,6,4,2]
mean = [0.3092317192600078, -1.9152401824371121, 4.407715829976413, 9.907990572810046, 0]
std = [-3.2761774178918777, -14.287992597889827, -3.7634234974064174, 44.105990175313984, -6.3244402096236305]
plotSize = (6 * 1.618, 6)
fig, ax = plt.subplots(figsize=plotSize)

ax.plot(x, mean, color=cmap(0), label = r"$\mu$")
ax1 = plt.twinx(ax)
ax1.invert_yaxis()
ax1.plot(x,std,color=cmap(2),label = r"$\sigma$")

ax.set_ylabel(r"$\Delta \mu$[%]")
ax1.set_ylabel(r"$\Delta \sigma$[%]")

ax.set_xlabel("Available classes")


ax.grid(True)
fig.legend(loc=4, prop={'size': 16})
plt.tight_layout()

plt.savefig("uni_comp.pdf")
plt.close()

