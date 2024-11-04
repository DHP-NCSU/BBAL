import matplotlib.pyplot as plt


baseline_dict = {29: 0.0548, 172: 0.0816, 58: 0.125, 41: 0.1317, 124: 0.169}
current_dict = {29: 0.1405, 172: 0.1633, 58: 0.3393, 41: 0.3417, 124: 0.2143}
# baseline_classes = sorted(list(baseline_dict.keys()))
baseline_classes = list(baseline_dict.keys())
current_classes = baseline_classes
baseline_accuracies = []
current_accuracies = []
for c in baseline_classes:
    baseline_accuracies.append(baseline_dict[c])
    current_accuracies.append(current_dict[c]-baseline_dict[c])
    
    
# Plot setup with overlapping bars for each class
plt.figure(figsize=(10, 5))

# Plotting baseline and current data in a single bar for each class with distinct color segments
bar_width = 0.4  # Adjusted width to show both in the same bar
index = range(len(baseline_classes))

# Plotting the baseline and current accuracies as stacked bars for each class
plt.bar(index, baseline_accuracies, width=bar_width, color='blue', label='Baseline')
plt.bar(index, current_accuracies, width=bar_width, color='orange', label='Current Acc', bottom=baseline_accuracies)

# Labels and title
plt.title('Baseline vs Worst Classes (Stacked)')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.xticks(index, baseline_classes)
plt.ylim(0, 0.5)
plt.grid(axis='y')
plt.legend()

# Adding annotation text within the plot
plt.text(len(baseline_classes) - 1, 0.45, r'$\lambda = 0.5, \gamma = 0.2$', 
         fontsize=12, verticalalignment='top', horizontalalignment='center')

# Adjust layout and display plot
plt.tight_layout()
plt.show()
