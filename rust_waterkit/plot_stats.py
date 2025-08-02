import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stats = ['GCMC', 'GCMCMC', 'GCMCSA', 'WK']
    dfs = []
    cell_text = []
    # Create a 4x1 subplot grid
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))

    # Ensure axes is a list for iteration (in case of 1D subplot grid)
    # axes = [axes] if not isinstance(axes, list) else axes
        
    for algorithm in stats:
        df = pd.read_csv(f"stats_{algorithm}_MD.csv")
        dfs.append(df)
    
    # Loop through each DataFrame and subplot
    for i, (df, title, ax) in enumerate(zip(dfs, stats, axes)):
        # Convert DataFrame to a list of lists for the table (excluding index)
        table_data = [df.columns.values.tolist()] + df.values.tolist()
        
        # Create table in the subplot
        table = ax.table(cellText=table_data,
                        colLabels=None,  # Columns are already included in table_data
                        cellLoc='center',
                        loc='center')
        
        # Adjust table appearance
        table.auto_set_font_size(True)
        # table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust scaling for better readability
        
        # Set title for the subplot
        ax.set_title(title)
        
        # Hide axes for a clean look
        ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("table_stats.png", dpi=300)