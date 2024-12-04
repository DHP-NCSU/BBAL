import sys

far, far_sd = 0.0, 0.0
mf, mf_sd = 0.0, 0.0
mp, mp_sd = 0.0, 0.0
mr, mr_sd = 0.0, 0.0
wf = 0.0
wp = 0.0
wr = 0.0

config = ""

for line in sys.stdin:
    try:
        if line.strip().startswith("Config"):
            cf = line.strip().split(": ")[1]
            if cf[-2:] != "_1":
                print(f"\nConfig: {config}")
                print("Metric		AUC		Std Dev")
                print("----------------------------------------")
                print(f"false_alarm_rate\t{far/2:.8f}\t{far_sd/2:.8f}")
                print(f"macro_f1\t{mf/2:.8f}\t{mf_sd/2:.8f}")
                print(f"macro_precision\t{mp/2:.8f}\t{mp_sd/2:.8f}")
                print(f"macro_recall\t{mr/2:.8f}\t{mr_sd/2:.8f}")
                print(f"weighted_f1\t{wf/2:.8f}")
                print(f"weighted_precision\t{wp/2:.8f}")
                print(f"weighted_recall\t{wr/2:.8f}")
                config = cf
                far, far_sd = 0.0, 0.0
                mf, mf_sd = 0.0, 0.0
                mp, mp_sd = 0.0, 0.0
                mr, mr_sd = 0.0, 0.0
                wf = 0.0
                wp = 0.0
                wr = 0.0
        if line.strip().startswith("false_alarm_rate"):
            _, _far, _far_sd = line.strip().split('\t')
            far += float(_far)
            far_sd += float(_far_sd)
        if line.strip().startswith("macro_f1"):
            _, _, _mf, _mf_sd = line.strip().split('\t')
            mf += float(_mf)
            mf_sd += float(_mf_sd)
        if line.strip().startswith("macro_precision"):
            _, _mp, _mp_sd = line.strip().split('\t')
            mp += float(_mp)
            mp_sd += float(_mp_sd)
        if line.strip().startswith("macro_recall"):
            _, _, _mr, _mr_sd = line.strip().split('\t')
            mr += float(_mr)
            mr_sd += float(_mr_sd)
        if line.strip().startswith("weighted_f1"):
            wf += float(line.strip().split('\t')[2])
        if line.strip().startswith("weighted_precision"):
            wp += float(line.strip().split('\t')[1])
        if line.strip().startswith("weighted_recall"):
            wr += float(line.strip().split('\t')[1])
    except:
        print(line.strip().split('\t'))
        break
    
print(f"\nConfig: {config}")
print("Metric		AUC		Std Dev")
print("----------------------------------------")
print(f"false_alarm_rate\t{far/2:.8f}\t{far_sd/2:.8f}")
print(f"macro_f1\t{mf/2:.8f}\t{mf_sd/2:.8f}")
print(f"macro_precision\t{mp/2:.8f}\t{mp_sd/2:.8f}")
print(f"macro_recall\t{mr/2:.8f}\t{mr_sd/2:.8f}")
print(f"weighted_f1\t{wf/2:.8f}")
print(f"weighted_precision\t{wp/2:.8f}")
print(f"weighted_recall\t{wr/2:.8f}")