import matplotlib.pyplot as plt
import numpy as np

# with open('GT_FAST/01.txt', 'r') as f:    
    # GT_xpoints = np.zeros(248, dtype=float) #need to read the folder results1_1.txt to count how many lines it writes aka pictures in 248
    # GT_ypoints = np.zeros(248, dtype=float)
    # i = 0 
#Ground Truth  
with open('build/ground_truth.txt', 'r') as f:  
    num_lines = sum(1 for line in f)
    GT_xpoints = np.zeros(num_lines, dtype=float)
    GT_ypoints = np.zeros(num_lines, dtype=float) #In the case the file are different length and ground truth isn't filled.
    f.seek(0)  # Reset file pointer to the beginning
    i = 0

    for line in f:
        data = line.split(" ")  
        GT_xpoints[i] = float(data[0])
        GT_ypoints[i] = float(data[1])
        # GT_ypoints[i] = float(data[2])
        i += 1
# with open('GT_FAST/results1_1.txt', 'r') as f:
with open('build/results1_1.txt', 'r') as f:   
    # result_xpoints = np.zeros(248, dtype=float)
    # result_ypoints = np.zeros(248, dtype=float)
    # i = 0
    num_lines = sum(1 for line in f)
    result_xpoints = np.zeros(num_lines, dtype=float)
    result_ypoints = np.zeros(num_lines, dtype=float)
    f.seek(0)  # Reset file pointer to the beginning
    i = 0

    for line in f:
        data = line.split(" ")  
        result_xpoints[i] = float(data[0])
        # result_ypoints[i] = float(data[1])
        result_ypoints[i] = float(data[2])
        i += 1
# For the first line of both 01.txt and results1_1.txt, compare the difference of the x and y points.
# The difference creates a shift of the ground truth to have the same first line of the 01.txt.
# The shift difference is added to all the ground truth points and rewritten as the new ground truth.
# To then plot again

# Calculate the shift difference
x_shift = GT_xpoints[0] - result_xpoints[0]
y_shift = GT_ypoints[0] - result_ypoints[0]

# Apply the shift to the ground truth points
fast_xpoints_shifted = GT_xpoints - x_shift
fast_ypoints_shifted = GT_ypoints - y_shift

# # Plot the shifted ground truth and AGAST points
# plt.plot(fast_xpoints_shifted, fast_ypoints_shifted)
# plt.plot(result_xpoints, result_ypoints, color='red')
# plt.legend(['FAST (ground truth)', 'AGAST'], loc='upper left')
# plt.show()
# plt.plot(GT_xpoints, GT_ypoints)
# plt.plot(result_xpoints, result_ypoints, color='red')
# plt.legend(['FAST (ground truth)', 'AGAST'], loc='upper left')
# plt.show()
# Plot the shifted ground truth, FAST, and AGAST points
plt.plot(fast_xpoints_shifted, fast_ypoints_shifted, color='blue')
# plt.plot(GT_xpoints, GT_ypoints, color='green')
plt.plot(result_xpoints, result_ypoints, color='red')
# plt.legend(['Shifted Ground Truth', 'Ground Truth', 'Result'], loc='upper left')
plt.legend(['Shifted Ground Truth', 'Result'], loc='upper left')

# max_x = max(np.max(fast_xpoints_shifted), np.max(GT_xpoints), np.max(result_xpoints))
plt.xlim(-3, 0)
plt.ylim(-1, 4)

plt.show()

print("FAST xpoints shifted:", fast_xpoints_shifted)
print("FAST ypoints shifted:", fast_ypoints_shifted)
# print("FAST xpoints:", GT_xpoints)
# print("FAST ypoints:", GT_ypoints)
print("AGAST xpoints:", result_xpoints)
print("AGAST ypoints:", result_ypoints)