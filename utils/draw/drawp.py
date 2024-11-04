import matplotlib.pyplot as plt

# 数据集
baseline_classes = [29, 58, 124, 41, 198]
baseline_accuracies = [0.081, 0.0863, 0.1214, 0.1232, 0.1315]
current_classes = [29, 58, 124, 41, 198]
current_accuracies = [0.1405, 0.3393, 0.2143, 0.3417, 0.3061]

baseline_dict = {29: 0.0548, 172: 0.0816, 58: 0.125, 41: 0.1317, 124: 0.169}
current_dict = {29: 0.1405, 172: 0.1633, 58: 0.3393, 41: 0.3417, 124: 0.2143}
baseline_classes = sorted(list(baseline_dict.keys()))
current_classes = baseline_classes
baseline_accuracies = []
current_accuracies = []
for c in baseline_classes:
    baseline_accuracies.append(baseline_dict[c])
    current_accuracies.append(current_dict[c])

# 画布设置
plt.figure(figsize=(10, 5))

# 画 baseline 和 current 数据
plt.plot(baseline_classes, baseline_accuracies, linestyle='--', marker='o', color='blue', label='Baseline')
plt.plot(current_classes, current_accuracies, linestyle='-', marker='o', color='orange', label='Current Acc')

# 图像标题、轴标签和标注
plt.title('Baseline vs Worst Classes')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.xticks(current_classes)
plt.ylim(0, 0.5)
plt.grid()
plt.legend()

# 添加文本标注，并将位置调整到图像范围内
plt.text(120, 0.45, r'$\lambda = 0.5, \gamma = 0.2$', fontsize=12, verticalalignment='top', horizontalalignment='center')

# 布局调整和展示图表
plt.tight_layout()
plt.show()
