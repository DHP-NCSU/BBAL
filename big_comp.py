import sys
import re
import ast

with open("comp5.log", 'r') as f:
    lines = f.readlines()
    
bla, bga, bdiff, bclass = 0.0, 0.0, 0.0, 0
la, ga = 0.0, 0.0
for line in lines:
    if "logs/gs_" in line:
        info = line.strip().split('_')
        la, ga = info[1], info[2][:-4]
    if "acc diff on 5 worst classes" not in line:
        continue
    dict_str = re.search(r'\{.*\}', line).group()
    pca = ast.literal_eval(dict_str)
    for k in pca.keys():
        if pca[k] > bdiff:
            bla, bga, bdiff, bclass = la, ga, pca[k], k
print(bla, bga, bclass, bdiff)
    
    