import pandas as pd
import os
import matplotlib.pyplot as plt

# Define the path to the results CSV file
results_file_path = os.path.join('c:/Users/pa112h/Documents/PhD_1/LysoTracker/241120_confocal/', 'Manders_average_df.csv')

# Load the results dataframe
results_df = pd.read_csv(results_file_path)

# Define the x-ticks
x_ticks = range(len(results_df['Time']))

plt.figure(figsize=(10, 6))
plt.errorbar(x_ticks, results_df['mM2_avg'], yerr=results_df['mM2_std'], linestyle='', marker='_', capsize=5)
plt.xlabel('Time (hours)')
plt.ylabel('Manders Coefficient')
plt.title('Overlap of NDs with Lysosomes')
plt.xticks(ticks=x_ticks, labels=[str(tick) for tick in results_df['Time']])
plt.show()
