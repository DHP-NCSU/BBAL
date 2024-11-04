import sys
import re
import ast

with open("comp5.log", 'r') as f:
    lines = f.readlines()
    
baseline = {29: 0.0548, 172: 0.0816, 58: 0.125, 41: 0.1317, 124: 0.169}
la, ga = 0.0, 0.0
for line in lines:
    if "logs/gs_" in line:
        info = line.strip().split('_')
        la, ga = info[1], info[2][:-4]
    if "current acc on 5 worst classes" not in line:
        continue
    if la == '15e' or la == '0.0' or ga == '0.0':
        continue
    dict_str = re.search(r'\{.*\}', line).group()
    pca = ast.literal_eval(dict_str)
    flag = True
    for k in pca.keys():
        if pca[k] <= baseline[k]:
            flag = False
    if flag:
        # print(f"\\{'textbf{'}{la}{'} & '}\\{'textbf{'}{ga}{'}'}", end='')
        print(f"*{la} & {ga}", end='')
    else:
        print(f"{la} & {ga}", end='')
    for k in pca.keys():
        print(f" & {round(pca[k]*100, 2)}\\%", end='')
    # print(f"{la} & {ga} & {round(pca[29]*100, 2)}\\% & {round(pca[172]*100, 2)}\\% & {round(pca[58]*100, 2)}\\% & {round(pca[41]*100, 2)}\\% & {round(pca[124]*100, 2)}\\% \\\\")
    print(" \\\\")