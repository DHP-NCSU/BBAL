import sys

acc_acc, acc_sd = 0.0, 0.0
cnt = 0

for line in sys.stdin:
    line = line.strip()
    if line[0] == '=':
        if "accuracy" in line:
            acc_acc += float(line.split(": ")[1][:-1]) / 100
            cnt += 1
        if "sd of accuracies" in line:
            acc_sd += float(line.split(": ")[1])
    else:
        acc, sd, _ = line.split(', ', 2)
        acc_acc += float(acc.split(': ')[1])
        acc_sd += float(sd.split(': ')[1])
        cnt += 1

assert(cnt == 21)
print(acc_acc / cnt)
print(acc_sd / cnt)
