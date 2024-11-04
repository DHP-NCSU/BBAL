import matplotlib.pyplot as plt

# Updated example accuracy data for two models (replace with your actual data)
iterations = list(range(1, 21))  # Iteration range from 0 to 21

with open("logs/baseline5.log", 'r') as f:
    lines = f.readlines()

accuracy_baseline = []
for line in lines:
    # if "====> Initial accuracy:" in line:
    #     accuracy_baseline.append(float(line.strip().split(': ')[1][:-1])/100)
    infos = line.split(', ')
    if len(infos) > 100:
        if line[0] == '=':
            continue
        accuracy_baseline.append(float(infos[1].split(': ')[1]))
import sys
accuracy_gs = []
for line in sys.stdin:
    # if "====> Initial accuracy:" in line:
    #     accuracy_gs.append(float(line.strip().split(': ')[1][:-1])/100)
    infos = line.split(', ')
    if len(infos) > 100:
        if line[0] == '=':
            continue
        accuracy_gs.append(float(infos[1].split(': ')[1]))
print(accuracy_gs)
# accuracy_baseline = [0.48, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.56, 0.57, 0.58, 0.58, 0.59, 0.59, 0.60, 0.61, 0.61, 0.62, 0.62, 0.63, 0.63, 0.63, 0.63]
# accuracy_gs = [0.48, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.58, 0.59, 0.60, 0.60, 0.61, 0.61, 0.62, 0.62, 0.63, 0.63, 0.64, 0.64, 0.64, 0.64]

plt.figure(figsize=(10, 5))
plt.plot(iterations, accuracy_baseline, 'o-', color='red', label='Standard Deviation of Per-Class Accuracy of baseline')
plt.plot(iterations, accuracy_gs, 'o-', color='blue', label=r'Standard Deviation of Per-Class Accuracy when $\lambda = 3.0, \gamma = 1.0$')

# Adding titles and labels
# plt.title('Accuracy Comparison: baseline5 vs gs_3.0_1.0', fontsize=15, color='darkblue', weight='bold')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('SD of Per-Class Acc', fontsize=12)


# Setting axis limits
plt.ylim(min(min(accuracy_baseline), min(accuracy_gs))-0.01, max(max(accuracy_gs), max(accuracy_baseline))+0.01)
plt.xlim(0, 21)

# Enabling grid
plt.grid(True)

# Adding legend
plt.legend()

# Set x-axis ticks to show each iteration
plt.xticks(range(1, 22, 1))  # This will show every integer from 0 to 21 on the x-axis

# Show plot
plt.show()
