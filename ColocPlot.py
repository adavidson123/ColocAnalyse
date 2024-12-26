import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"

root = tk.Tk()
root.withdraw()

#open GUI to ask to open folder path and select folder to analyse
folder_path = filedialog.askdirectory()

extension = "*.csv" #folder extension of output from ImageJ Log files

files = glob.glob(f"{folder_path}/{extension}", recursive=False)

t_afterNDs = []
tM1_avg = []
tM1_std = []
tM1_err = []
tM2_avg = []
tM2_std = []
tM2_err = []
tot_avg = []
tot_std = []
tot_err = []
for file in files:

    fileName = file.split('/')[-1] #find filename from path
    time_csv = fileName.split('_')[-1][:-5] #find the time after ND removal from file name
    t_afterNDs.append(float(time_csv)) #append t to t_afterNDs list

    df = pd.read_csv(file, index_col=0)

    #find the tM1, tM2 and avg values in each timestamped dataframe
    tM1_avg_value = df.loc['avg'][0]
    tM2_avg_value = df.loc['avg'][1]
    tot_avg_value = df.loc['avg'][2]

    #append all values to the required lists as floats
    tM1_avg.append(float(tM1_avg_value))
    tM2_avg.append(float(tM2_avg_value))
    tot_avg.append(float(tot_avg_value))

    #find each std of tM1, tM2, avg for each timestamped dataframe
    tM1_std_value = float(df.loc['std'][0])
    tM2_std_value = float(df.loc['std'][1])
    tot_std_value = float(df.loc['std'][2])

    #append all values to the required lists as floats
    tM1_std.append(tM1_std_value)
    tM2_std.append(tM2_std_value)
    tot_std.append(tot_std_value)

    #convert std to std err
    no_trials = float(len(df.index) - 2)

    def std_error(n, sigma):
        sqrt_n = np.sqrt(n)
        err = sigma/sqrt_n
        return err
    
    tM1_std_err = std_error(no_trials, tM1_std_value)
    tM2_std_err = std_error(no_trials, tM2_std_value)
    tot_std_err = std_error(no_trials, tot_std_value)

    #append all errors to the required lists
    tM1_err.append(tM1_std_err)
    tM2_err.append(tM2_std_err)
    tot_err.append(tot_std_err)

############################################################################################
#save this data as a dataframe of times and tM1, tM2 and avg with std err for later plotting
############################################################################################

df = pd.DataFrame({'tM1': tM1_avg, 'tM1_std': tM1_std, 'tM1_err': tM1_err, 
                   'tM2': tM2_avg, 'tM2_std': tM2_std, 'tM2_err': tM2_err,
                   'avg': tot_avg, 'std': tot_std, 'err': tot_err})
df.index = t_afterNDs
df.sort_index(ascending=True, inplace=True)

date = folder_path.split('/')[-1].split('_')[0] #get date from name of folder path

#plot data from this date and save plot in the same folder

fig, ax = plt.subplots()
ax.errorbar(t_afterNDs, tot_avg, yerr=tot_err, label='Average', linestyle='', marker='x', 
            color='black', capsize=2)
ax.errorbar(t_afterNDs, tM1_avg, yerr=tM1_err, label='tM1', linestyle='', marker='x', 
            color='#1b9e77', capsize=2)
ax.errorbar(t_afterNDs, tM2_avg, yerr=tM2_err, label='tM2', linestyle='', marker='x', 
            color='#d95f02', capsize=2)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Manders Coefficient')
#ax.set_xscale('log')
fig.legend(bbox_to_anchor = [0.6, 0.95], 
           frameon=False)
fig.tight_layout()

## make new folder to save these files to
new_path = f'{folder_path}/Analysis'

if not os.path.exists(new_path):
    os.mkdir(new_path)
    print(f"Folder '{new_path}' created.")
else:
    print(f"Folder '{new_path}' already exists.")

#save these plots to the same folder
plt.savefig(f"{folder_path}/Analysis/{date}_TimeColocPlot.png")

df.to_csv(f'{folder_path}/Analysis/{date}_TimeColocData.csv')

#plt.show()
