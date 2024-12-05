import sys
import re

def format_sci(num):
    # Format number as x.xxe±y
    exp = f"{num:.3e}"  # Get basic scientific notation
    # Split into coefficient and exponent
    coeff, e = exp.split('e')
    # Format exponent to remove leading zeros and + sign
    e = int(e)
    return f"{coeff}e{e}"

def parse_metrics(text):
    # Extract lambda and gamma from Config line
    config_match = re.search(r'gs_(\d+\.\d+)_(\d+\.\d+)', text)
    if config_match:
        lambda_val = float(config_match.group(1))
        gamma_val = float(config_match.group(2))
    else:
        lambda_val = gamma_val = float('nan')
    
    # Initialize metrics dictionary
    metrics = {
        'far': (0.0, 0.0),
        'f1': (0.0, 0.0),
        'precision': (0.0, 0.0),
        'recall': (0.0, 0.0)
    }
    
    # Parse metrics
    lines = text.split('\n')
    for line in lines:
        if 'false_alarm_rate' in line:
            vals = line.split('\t')
            metrics['far'] = (float(vals[1]), float(vals[2]))
        elif 'macro_f1' in line:
            vals = line.split('\t')
            metrics['f1'] = (float(vals[1]), float(vals[2]))
        elif 'macro_precision' in line:
            vals = line.split('\t')
            metrics['precision'] = (float(vals[1]), float(vals[2]))
        elif 'macro_recall' in line:
            vals = line.split('\t')
            metrics['recall'] = (float(vals[1]), float(vals[2]))
    
    return lambda_val, gamma_val, metrics

def format_latex_row(lambda_val, gamma_val, metrics):
    return f"{lambda_val:.1f} & {gamma_val:.1f} & " + \
           f"{format_sci(metrics['far'][0])} & {format_sci(metrics['far'][1])} & " + \
           f"{format_sci(metrics['f1'][0])} & {format_sci(metrics['f1'][1])} & " + \
           f"{format_sci(metrics['precision'][0])} & {format_sci(metrics['precision'][1])} & " + \
           f"{format_sci(metrics['recall'][0])} & {format_sci(metrics['recall'][1])} \\\\"

# Print LaTeX table header
print("\\begin{tabular}{llllllllll}")
print("\\hline")
print("$\\lambda$ & $\\gamma$ & FAR & sdFAR & F1 & sdF1 & Prec & sdPrec & Recall & sdRecall \\\\")
print("\\hline")

# Read and process input
current_block = []
for line in sys.stdin:
    line = line.strip()
    if line:
        current_block.append(line)
    else:
        if current_block:
            text = '\n'.join(current_block)
            lambda_val, gamma_val, metrics = parse_metrics(text)
            print(format_latex_row(lambda_val, gamma_val, metrics))
            current_block = []

# Process last block if exists
if current_block:
    text = '\n'.join(current_block)
    lambda_val, gamma_val, metrics = parse_metrics(text)
    print(format_latex_row(lambda_val, gamma_val, metrics))

# Print table footer
print("\\hline")
print("\\end{tabular}")