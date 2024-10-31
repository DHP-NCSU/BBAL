import sys
import re
import ast

with open("comp.log", 'r') as f:
    lines = f.readlines()
    
la, ga = 0.0, 0.0
for line in lines:
    if "logs/gs_" in line:
        info = line.strip().split('_')
        la, ga = info[1], info[2][:-4]
    if "current acc on 5 worst classes" not in line:
        continue
    dict_str = re.search(r'\{.*\}', line).group()
    pca = ast.literal_eval(dict_str)
    print(f"{la} & {ga} & {round(pca[29]*100, 2)}\\% & {round(pca[58]*100, 2)}\\% & {round(pca[124]*100, 2)}\\% & {round(pca[41]*100, 2)}\\% & {round(pca[198]*100, 2)}\\% \\\\")
    