import re
import matplotlib.pyplot as plt
import os

data = {}

current_iteration = None
log_files = ["logs//baseline5.log", "logs//gs_3.0_1.0.log"]


color_map = {
    "logs//baseline5.log": "crimson",
    "logs//gs_3.0_1.0.log": "navy",
    
}

for log_file in log_files:
    iterations = []
    accuracies = []
    sd_values = []
    
    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            iteration_pattern = r"Iteration:\s(\d+)"
            iteration_match = re.search(iteration_pattern, line)
            
            acc_pattern = r"acc:\s([0-9.]+),\ssd of accuracies:\s([0-9.]+)"
            acc_match = re.search(acc_pattern, line)
            
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                print(f"Found Iteration: {current_iteration}")
            
            elif acc_match and current_iteration is not None:
                acc = float(acc_match.group(1))
                sd_of_accuracies = float(acc_match.group(2))
                
                iterations.append(current_iteration)
                accuracies.append(acc)
                sd_values.append(sd_of_accuracies)

                print(f"Match found - Iteration: {current_iteration}, Accuracy: {acc}, SD: {sd_of_accuracies}")
                
                current_iteration = None
            else:
                print(f"No match found in line: {line.strip()}")  

    data[log_file] = {
        'iteration': iterations,
        'accuracies': accuracies,
        'sd_values': sd_values,
    }


plt.figure(figsize=(10, 5))
plt.plot(data["logs//baseline5.log"]['iteration'], data["logs//baseline5.log"]['accuracies'], 
         marker="o", label="Accuracy from baseline5", color=color_map["logs//baseline5.log"])
plt.plot(data["logs//gs_3.0_1.0.log"]['iteration'], data["logs//gs_3.0_1.0.log"]['accuracies'], 
         marker="o", label="Accuracy from gs_3.0_1.0", color=color_map["logs//gs_3.0_1.0.log"])


plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Accuracy Comparison: baseline5 vs gs_3.0_1.0", fontsize=16, weight='bold', color='darkblue')
plt.xticks(range(1, max(data["logs//baseline5.log"]['iteration']) + 1))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("pic/acc baseline vs 3_1.png", format='png', dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
for log_file in log_files:
    iterations = data[log_file]['iteration']
    sd_values = data[log_file]['sd_values']
    plt.plot(iterations, sd_values, marker="x", label=f"SD from {os.path.splitext(os.path.basename(log_file))[0]}", color=color_map[log_file])

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Standard Deviation", fontsize=14)
plt.title("Standard Deviation of Accuracy Over Iterations", fontsize=16, weight='bold', color='darkblue')
plt.xticks(range(1, max([max(data[log_file]['iteration']) for log_file in log_files]) + 1))
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("pic/sd baseline vs 3_1.png", format='png', dpi=300)
plt.show()



'''
    
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, accuracies, label="Accuracy", marker="o", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Iterations",)
    plt.xticks(ticks=range(min(iterations), max(iterations) + 1))
    plt.grid()
    plt.show()
    # 绘制 sd 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, sd_values, label="Standard Deviation", marker="x", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation over Iterations")
    plt.xticks(ticks=range(min(iterations), max(iterations) + 1))
    plt.grid()
    plt.show()
    '''
    # 绘制 accuracy 曲线
'''plt.plot(iterations, accuracies, label="Accuracy", marker="o")
    plt.plot(iterations, sd_values, label="Standard Deviation", marker="x")

    # 添加图例、标题和标签
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Accuracy and SD over Iterations")
    plt.legend()
    plt.grid()

    # 显示图形
    plt.show()
    '''
#else:
    #print("No data extracted. Please check the log file format and regex pattern.")
