import matplotlib.pyplot as plt
import numpy as np

with open('GT_FAST/01.txt', 'r') as f:    
    # fast_xpoints = np.zeros(248, dtype=float) #need to read the folder results1_1.txt to count how many lines it writes aka pictures in 248
    # fast_ypoints = np.zeros(248, dtype=float)
    # i = 0
    num_lines = sum(1 for line in f)
    fast_xpoints = np.zeros(num_lines, dtype=float)
    fast_ypoints = np.zeros(num_lines, dtype=float) #In the case the file are different length and ground truth isn't filled.
    f.seek(0)  # Reset file pointer to the beginning
    i = 0

    for line in f:
        data = line.split(" ")  
        fast_xpoints[i] = float(data[0])
        fast_ypoints[i] = float(data[1])
        # fast_ypoints[i] = float(data[2])
        i += 1

with open('GT_FAST/results1_1.txt', 'r') as f:    
    # agast_xpoints = np.zeros(248, dtype=float)
    # agast_ypoints = np.zeros(248, dtype=float)
    # i = 0
    num_lines = sum(1 for line in f)
    agast_xpoints = np.zeros(num_lines, dtype=float)
    agast_ypoints = np.zeros(num_lines, dtype=float)
    f.seek(0)  # Reset file pointer to the beginning
    i = 0

    for line in f:
        data = line.split(" ")  
        agast_xpoints[i] = float(data[0])
        agast_ypoints[i] = float(data[1])
        # agast_ypoints[i] = float(data[2])
        i += 1
# For the first line of both 01.txt and results1_1.txt, compare the difference of the x and y points.
# The difference creates a shift of the ground truth to have the same first line of the 01.txt.
# The shift difference is added to all the ground truth points and rewritten as the new ground truth.
# To then plot again

# Calculate the shift difference
x_shift = fast_xpoints[0] - agast_xpoints[0]
y_shift = fast_ypoints[0] - agast_ypoints[0]

# Apply the shift to the ground truth points
fast_xpoints_shifted = fast_xpoints - x_shift
fast_ypoints_shifted = fast_ypoints - y_shift

# # Plot the shifted ground truth and AGAST points
# plt.plot(fast_xpoints_shifted, fast_ypoints_shifted)
# plt.plot(agast_xpoints, agast_ypoints, color='red')
# plt.legend(['FAST (ground truth)', 'AGAST'], loc='upper left')
# plt.show()
# plt.plot(fast_xpoints, fast_ypoints)
# plt.plot(agast_xpoints, agast_ypoints, color='red')
# plt.legend(['FAST (ground truth)', 'AGAST'], loc='upper left')
# plt.show()
# Plot the shifted ground truth, FAST, and AGAST points
plt.plot(fast_xpoints_shifted, fast_ypoints_shifted, color='blue')
plt.plot(fast_xpoints, fast_ypoints, color='green')
plt.plot(agast_xpoints, agast_ypoints, color='red')
plt.legend(['FAST (shifted)', 'FAST', 'AGAST'], loc='upper left')
plt.show()