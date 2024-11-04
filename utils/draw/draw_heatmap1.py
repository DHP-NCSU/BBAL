import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors

# λ 和 γ
lambda_values = [0, 0.05, 0.1, 0.2, 0.5, 1, 3]
gamma_values = [0, 0.1, 0.2, 1, 3, 5]

# load data (example data handling)
with open("cal.log", 'r') as f:
    lines = f.readlines()

data = {}
la, ga = 0.0, 0.0
for line in lines:
    if "logs" in line:
        infos = line.strip().split('_')
        la = float(infos[1])
        ga = float(infos[2].replace(".log", ''))
        continue
    if line.strip() == '':
        continue
    val = float(line.strip())
    if val > 0.3:
        data[(la, ga)] = val

data[(0.0, 0.0)] = 0.5778

# Create DataFrame
df = pd.DataFrame(index=lambda_values, columns=gamma_values)
for (lam, gam), acc in data.items():
    df.at[lam, gam] = acc
df = df.astype(float)

plt.figure(figsize=(10, 8))

# Set custom color map from light to deep blue
cmap = sns.color_palette("Blues", as_cmap=True)

# Create heatmap
heatmap = sns.heatmap(df, annot=False, cmap=cmap, cbar_kws={'label': 'Accuracy'},
                      fmt="", linewidths=0.5, linecolor='gray',
                      vmin=df.min().min(), vmax=df.max().max(), mask=df.isnull())

# Function to determine text color based on background luminance
def get_text_color(background_color):
    r, g, b = background_color[:3]  # RGB values
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.5 else 'black'

# Use colormap to get colors based on normalized values
norm = mcolors.Normalize(vmin=df.min().min(), vmax=df.max().max())

# Place values with color based on background
for lam in lambda_values:
    for gam in gamma_values:
        if not pd.isna(df.at[lam, gam]):  # process only the normal area
            x_idx = gamma_values.index(gam) + 0.5  # Centering
            y_idx = lambda_values.index(lam) + 0.5
            value = df.at[lam, gam]
            color = cmap(norm(value))  # Get color from colormap based on normalized value
            text_color = get_text_color(color)
            heatmap.text(x=x_idx, y=y_idx, s=f"{value:.4f}",
                         color=text_color, fontsize=12, ha='center', va='center')

# Set labels
plt.xlabel(r'$\gamma$', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)

# Show plot
plt.show()
