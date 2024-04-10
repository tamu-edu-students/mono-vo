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
#scale GT points to have it match the result points

# print("GT xpoints:", GT_xpoints)
# print("GT ypoints:", GT_ypoints)
# # Calculate the scaling factor
# x_scale = np.mean(result_xpoints) / np.mean(GT_xpoints)
# y_scale = np.mean(result_ypoints) / np.mean(GT_ypoints)
# # GT_xpoints = GT_xpoints * x_scale
# # GT_ypoints = GT_ypoints * y_scale

# # Calculate the shift difference
# x_shift = GT_xpoints[0]*x_scale - result_xpoints[0]
# y_shift = GT_ypoints[0]*y_scale - result_ypoints[0]

# # Apply the shift to the ground truth points
# GT_xpoints_shift = GT_xpoints*x_scale - x_shift
# GT_ypoints_shift = GT_ypoints*y_scale - y_shift

# Calculate the scaling factor based on range
GT_ypoints = -GT_ypoints #from my observation, the ground truth needs to be flipped for y axis
x_scale = (np.max(result_xpoints) - np.min(result_xpoints)) / (np.max(GT_xpoints) - np.min(GT_xpoints))
y_scale = (np.max(result_ypoints) - np.min(result_ypoints)) / (np.max(GT_ypoints) - np.min(GT_ypoints))

# Scale the GT points
GT_xpoints_scaled = (GT_xpoints - np.min(GT_xpoints)) * x_scale + np.min(result_xpoints)
GT_ypoints_scaled = (GT_ypoints - np.min(GT_ypoints)) * y_scale + np.min(result_ypoints)

#after we scale and shift, let's make sure that we have the right direction. Do a rotation or reflection to reduce residual error
# Calculate the shift difference
print("GT xpoints shifted and scaled:", GT_xpoints_scaled)
print("GT ypoints shifted and scaled:", GT_ypoints_scaled)
# print("FAST xpoints:", GT_xpoints)
# print("FAST ypoints:", GT_ypoints)
print("Result xpoints:", result_xpoints)
print("Result ypoints:", result_ypoints)

# # Plot the shifted ground truth and AGAST points
# plt.plot(GT_xpoints_shift, GT_ypoints_shift)
# plt.plot(result_xpoints, result_ypoints, color='red')
# plt.legend(['FAST (ground truth)', 'AGAST'], loc='upper left')
# plt.show()
# plt.plot(GT_xpoints, GT_ypoints)
# plt.plot(result_xpoints, result_ypoints, color='red')
# plt.legend(['FAST (ground truth)', 'AGAST'], loc='upper left')
# plt.show()
# Plot the shifted ground truth, FAST, and AGAST points
plt.plot(GT_xpoints_scaled, GT_ypoints_scaled, color='blue')
# plt.plot(GT_xpoints, GT_ypoints, color='green')
plt.plot(result_xpoints, result_ypoints, color='red')
# plt.legend(['Shifted Ground Truth', 'Ground Truth', 'Result'], loc='upper left')
plt.legend(['Scaled Ground Truth', 'Result'], loc='upper left')

# max_x = max(np.max(GT_xpoints_shift), np.max(GT_xpoints), np.max(result_xpoints))
# plt.xlim(-3, 0)
# plt.ylim(-1, 4)

plt.show()