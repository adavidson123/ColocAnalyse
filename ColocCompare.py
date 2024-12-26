#input in files from Analysis folders and make two plots
# - one of each tM1, tM2 and avg with error bars for each data set from different days
# - one with average tM1, tM2 and avg with error bars for all different day data sets

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"

root = tk.Tk()
root.withdraw()

def read_analysis_csv(global_dir):
    # Initialize an empty list to store the DataFrames
    dataframes = []
    dataframes_2 = []
    
    # Walk through the global directory to find 'Analysis' subdirectory
    for root, dirs, files in os.walk(global_dir):
        # Check if 'Analysis' is in the current directory path
        if 'Analysis' in dirs:
            analysis_dir = os.path.join(root, 'Analysis')  # Full path to 'Analysis' subdirectory
            # List all files in the 'Analysis' subdirectory
            for file in os.listdir(analysis_dir):
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(analysis_dir, file)

                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(csv_file_path, index_col=0)
                    # Append the DataFrame to the list
                    dataframes.append(df)
                    dataframes_2.append(df[['tM1_std', 'tM1_err', 'tM2_std', 'tM2_err',
                                            'std', 'err']])

    return dataframes, dataframes_2

#using the above function with the interactive file choosing GUI
#open GUI to ask to open folder path and select folder to analyse - select the global folder with all
#dates in as this code will compare each different date
global_directory = filedialog.askdirectory()
csv_dataframes, error_dataframes = read_analysis_csv(global_directory)

#print(csv_dataframes)
#print(error_dataframes)


# Concatenate all DataFrames along the rows (axis=0) and then compute the mean along rows (axis=0)
#################################################################################################
# may not be combining errors correctly here - check this and change maybe
#################################################################################################
# make a different dataframes for the errors and find the mean error...???
##################################################################################################

#compute the mean of the Manders coefficients from the original data
average_df = pd.concat(csv_dataframes).groupby(level=0).mean()

#calculate the number of samples for each Manders coefficient (std/err)^2 = n
#propagate error from 1/n sqrt(sum (err^2))

n = len(error_dataframes) #assumes that each dataset has all the same time stamps

for dataframe in error_dataframes:
    dataframe['tM1_sq'] = dataframe['tM1_err'] ** 2 
    dataframe['tM2_sq'] = dataframe['tM2_err'] ** 2
    #print(dataframe)

prop_error_df = pd.concat(error_dataframes).groupby(level=0).sum()

prop_error_df['tM1_new_err'] = (prop_error_df['tM1_sq']) ** (1/2) / n
prop_error_df['tM2_new_err'] = (prop_error_df['tM2_sq'] ** (1/2)) / n

#reset error in average_df tM1_new_err and tM2_new_err to average_df dataframe

average_df['tM1_err'] = prop_error_df['tM1_new_err']
average_df['tM2_err'] = prop_error_df['tM2_new_err']

#remove avg, std and err

average_df.drop(columns=['avg', 'std', 'err', 'tM1_std', 'tM2_std'], inplace=True)
print(average_df)

#mplotting averages of each M1, M2 and avg across all the trials
fig2, ax2 = plt.subplots()
ax2.errorbar(average_df.index, average_df.loc[:,'tM1'], yerr=average_df.loc[:, 'tM1_err'],
             linestyle='', marker='x', capsize=2, label='M1', color='#1b9e77')
ax2.errorbar(average_df.index, average_df.loc[:,'tM2'], yerr=average_df.loc[:, 'tM2_err'],
             linestyle='', marker='x', capsize=2, label='M2', color='#d95f02')
#ax2.errorbar(average_df.index, average_df.loc[:,'avg'], yerr=average_df.loc[:, 'err'],
 #            linestyle='', marker='x', capsize=2, label='Average', color='black')

ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Manders Coefficient')
fig2.legend(bbox_to_anchor = [0.6, 0.95], 
           frameon=False)
fig2.tight_layout()

#different subplots on the same axis of tM1, tM2, avg for each experiment
#plotting of data from csv_dataframes
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))
for i, df in enumerate(csv_dataframes):

    #locate each of the values for plotting from each element in the list of dataframes
    tM1_i = csv_dataframes[i].loc[:,'tM1']
    tM1_err_i = csv_dataframes[i].loc[:, 'tM1_err']
    tM2_i = csv_dataframes[i].loc[:, 'tM2']
    tM2_err_i = csv_dataframes[i].loc[:, 'tM2_err']
    #avg_i = csv_dataframes[i].loc[:, 'avg']
    #err_i = csv_dataframes[i].loc[:, 'err']

    # add in here locating the date the experiment was done to add to a legend??

    #plot each element against the time index for each dataframe in the list
    ax[0].errorbar(tM1_i.index, tM1_i, yerr=tM1_err_i, linestyle='', marker='x', capsize=2)
    ax[1].errorbar(tM2_i.index, tM2_i, yerr=tM2_err_i, linestyle='', marker='x', capsize=2)
    #ax[2].errorbar(avg_i.index, avg_i, yerr=err_i, linestyle='', marker='x', capsize=2)

ax[0].errorbar(average_df.index, average_df.loc[:,'tM1'], yerr=average_df.loc[:,'tM1_err'], 
               linestyle='', marker='x', capsize=2, color='black')
ax[0].set_ylabel('Manders Coefficient')
ax[0].set_title('M1')
ax[0].set_xlabel('Time (h)')
ax[1].errorbar(average_df.index, average_df.loc[:,'tM2'], yerr=average_df.loc[:,'tM2_err'], 
               linestyle='', marker='x', capsize=2, color='black')
ax[1].set_title('M2')
ax[1].set_xlabel('Time (h)')
#ax[2].errorbar(average_df.index, average_df.loc[:,'avg'], yerr=average_df.loc[:,'err'], 
              # linestyle='', marker='x', capsize=2, color='black')
#ax[2].set_title('Average')
#ax[2].set_xlabel('Time (h)')
fig.tight_layout()


#save figure into a created folder at the end
## make new folder to save these files to
new_path = f'{global_directory}/AllDataPlots'

if not os.path.exists(new_path):
    os.mkdir(new_path)
    print(f"Folder '{new_path}' created.")
else:
    print(f"Folder '{new_path}' already exists.")

#save these plots to the same folder
fig.savefig(f"{new_path}/SubPlots.png")
fig2.savefig(f"{new_path}/AveragedData.png")

#save average_df dataframe to csv file in case wanted for future plotting
df.to_csv(f'{new_path}/AveragedData.csv')

plt.show()

###################################################################################################
# add a graph comparing the average M1 and M2 with the simulated values of M1 and M2
##################################################################################################