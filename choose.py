import re

# 设置日志文件路径
log_file_path = "/root/CEAL/comp.log"

# 用于记录最大平均变化信息
max_avg_diff = -float('inf')
best_diff_line = ''
best_lambda = None
best_gamma = None

# 定义匹配文件名的正则表达式
log_pattern = r'gs_(\d+\.?\d*)_(\d+\.?\d*).log'

# 打开日志文件
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        # 查找包含文件名的行
        match = re.search(r'logs/gs_(\d+\.?\d*)_(\d+\.?\d*).log', line)
        if match:
            # 提取参数值
            lambda_value = match.group(1)
            gamma_value = match.group(2)

            # 继续读取接下来的行以找到相关的数据
            for _ in range(4):  # 读取接下来的4行
                line = next(log_file)

                # 只提取包含"acc diff on 5 worst classes"的行
                if "acc diff on 5 worst classes" in line:
                    # 提取每个类别的变化值
                    diffs = [float(value) for value in re.findall(r'(\d+\.\d+)', line)]
                    if diffs:  # 确保有提取到变化值
                        avg_diff = sum(diffs) / len(diffs)  # 计算平均变化值
                        if avg_diff > max_avg_diff:
                            max_avg_diff = avg_diff
                            best_diff_line = line.strip()
                            best_lambda = lambda_value
                            best_gamma = gamma_value

# 输出结果
print(f"对应行: {best_diff_line}")
print(f"最大平均变化值: {max_avg_diff}")
print(f"对应的参数: λ = {best_lambda}, γ = {best_gamma}")

