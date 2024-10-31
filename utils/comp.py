import sys
import re
import ast

COMP_NUM = 5 
DICIMAL = 4
with open("./logs/gs_0.0_0.0.log", 'r') as f:
    lines = f.readlines()
    
baseline_acc = {i: 0.0 for i in range(256)}
for line in lines:
    if "per-class accuracies" not in line:
        continue
    dict_str = re.search(r'\{.*\}', line).group()
    pca = ast.literal_eval(dict_str)
    for i in range(256):
        baseline_acc[i] += pca[i] / 21

bad_keys = sorted(baseline_acc, key=baseline_acc.get)[:COMP_NUM]
bad = {key: round(baseline_acc[key], DICIMAL) for key in bad_keys}
print(f"worst {COMP_NUM} classes in baseline:     {bad}")

current_acc = {i: 0.0 for i in range(256)}
for line in sys.stdin:
    if "per-class accuracies" not in line:
        continue
    dict_str = re.search(r'\{.*\}', line).group()
    pca = ast.literal_eval(dict_str)
    for i in range(256):
        current_acc[i] += pca[i] / 21

difference = {i: round(current_acc[i] - baseline_acc[i], DICIMAL) for i in range(256)}
current_bad = {key: round(current_acc[key], DICIMAL) for key in bad_keys}
diff_bad = {key: difference[key] for key in bad_keys}
print(f"current acc on {COMP_NUM} worst classes:  {current_bad}")
print(f"acc diff on {COMP_NUM} worst classes:     {diff_bad}")
diff_keys = sorted(difference, key=difference.get, reverse=True)[:COMP_NUM]
diff = {key: difference[key] for key in diff_keys}
print(f"{COMP_NUM} classes with largest diff:     {diff}")
improvement = {i: round(difference[i] / baseline_acc[i], DICIMAL) for i in range(256)}
impv_keys = sorted(improvement, key=improvement.get, reverse=True)[:COMP_NUM]
impv = {key: improvement[key] for key in impv_keys}
print(f"{COMP_NUM} classes with largest impv:     {impv}")

