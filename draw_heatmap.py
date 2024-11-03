import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#  λ 和 γ 
lambda_values = [0, 0.05, 0.1, 0.2, 0.5, 1, 3]
gamma_values = [0, 0.1, 0.2, 1, 3, 5]

#load data
data = {
    (0, 0): 0.5785, (0.05, 0.1): 0.5773, (0.05, 0.2): 0.5755,
    (0.1, 0.1): 0.5841, (0.1, 0.2): 0.5733, (0.2, 0.1): 0.5764,
    (0.2, 0.2): 0.5697, (0.5, 0.1): 0.5756, (0.5, 0.2): 0.5709,
    (0.5, 3): 0.5761, (0.5, 5): 0.5757, (1, 0): 0.577, 
    (1, 1): 0.5728, (1, 3): 0.5727, (1, 5): 0.58,
    (3, 0): 0.5777, (3, 1): 0.5837
}

#create DataFrame 
df = pd.DataFrame(index=lambda_values, columns=gamma_values)
for (lam, gam), acc in data.items():
    df.at[lam, gam] = acc


df = df.astype(float)


plt.figure(figsize=(10, 8))


cmap = sns.color_palette("YlGnBu", as_cmap=True)


heatmap = sns.heatmap(df, annot=False, cmap=cmap, cbar_kws={'label': 'Accuracy'}, 
                       fmt=".4f", linewidths=0.5, linecolor='gray', 
                       vmin=df.min().min(), vmax=df.max().max(), mask=df.isnull())


highlight_positions = {(3.0, 1.0), (0.5, 3.0), (1.0, 5.0)}
white_positions = {(0.1,0.1),(1.0,0.0),(3.0,0.0),(3.0,1.0),(1.0,5.0),(0.0,0.0),(0.05,0.1)}

for lam in lambda_values:
    for gam in gamma_values:
        x_idx = gamma_values.index(gam) + 0.5  # +0.5 centerize
        y_idx = lambda_values.index(lam) + 0.5 
        if(lam, gam) not in highlight_positions and not pd.isna(df.at[lam, gam]) and (lam, gam) not in white_positions:  # process only the normal area
            # display the value
                heatmap.text(x=x_idx, y=y_idx, s=f"{df.at[lam, gam]:.4f}", 
                          color='black', fontsize=10, ha='center', va='center')
        elif (lam, gam) not in highlight_positions:
            heatmap.text(x=x_idx, y=y_idx, s=f"{df.at[lam, gam]:.4f}",
                          color='white', fontsize=10, ha='center', va='center')
        

# highlight area 
for (lam, gam) in highlight_positions:
    x_idx = gamma_values.index(gam) + 0.5  
    y_idx = lambda_values.index(lam) + 0.5  
    
    # change the highlight area color
    heatmap.add_patch(plt.Rectangle((x_idx-0.5, y_idx-0.5), 1, 1, fill=True,alpha=0.5))

    heatmap.text(x=x_idx, y=y_idx, s=f"{df.at[lam, gam]:.4f}", 
                  color='#990000', fontsize=10, weight='bold', ha='center', va='center')

#set labels
plt.xlabel(r'$\gamma$', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=14)
plt.title("Model Accuracy for Different λ and γ Combinations", fontsize=18, weight='bold', color="darkblue")

# show
plt.show()




