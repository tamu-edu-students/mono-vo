import matplotlib.pyplot as plt
import numpy as np

with open('GT_FAST/01.txt', 'r') as f:    
    fast_xpoints = np.zeros(248, dtype=float)
    fast_ypoints = np.zeros(248, dtype=float)
    i = 0

    for line in f:
        data = line.split(" ")  
        fast_xpoints[i] = float(data[0])
        fast_ypoints[i] = float(data[2])
        i += 1

with open('GT_FAST/results1_1.txt', 'r') as f:    
    agast_xpoints = np.zeros(248, dtype=float)
    agast_ypoints = np.zeros(248, dtype=float)
    i = 0

    for line in f:
        data = line.split(" ")  
        agast_xpoints[i] = float(data[0])
        agast_ypoints[i] = float(data[2])
        i += 1

plt.plot(fast_xpoints, fast_ypoints)
plt.plot(agast_xpoints, agast_ypoints, color='red')
plt.show()
