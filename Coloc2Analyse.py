import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd

root = tk.Tk()
root.withdraw()

#open GUI to ask to open folder path and select folder to analyse
folder_path = filedialog.askdirectory()

extension = "*.txt" #folder extension of output from ImageJ Log files

files = glob.glob(f"{folder_path}/{extension}", recursive=False)

for file in files:
    
    df = pd.DataFrame() #make new dataframe
    infile = open(file, "r") #open each .txt file and read it
    dataLines = infile.readlines() #split .txt file into lines to read
    infile.close() #close the file but lines saved locally in dataLines
    
    #add name to df e.g. yymmdd_xh and add column titles of different Manders coeffs
    imageNames = []
    tM1 = []
    tM2 = []
    SplitName = dataLines[2].split(' ')[1][:-5]
    #add names of different images to imageNames list 
    for linenum, line in enumerate(dataLines):
        if "Processing image:" in line: #add image names to df
            imageName_line = dataLines[linenum]
            imageName_data = imageName_line.split(' ')[4].strip()
            imageNames.append(imageName_data)
        
        if "Manders' tM1 (Above autothreshold of Ch2)" in line: #add tM1 to df
            tM1_line = dataLines[linenum]
            tM1_data = float(tM1_line.split(' ')[6].strip())
            tM1.append(tM1_data)

        if "Manders' tM2 (Above autothreshold of Ch1)" in line: #add tM2 to df
            tM2_line = dataLines[linenum]
            tM2_data = float(tM2_line.split(' ')[6].strip())
            tM2.append(tM2_data)
        
    df[f'{SplitName}'] = imageNames
    df['tM1'] = tM1
    df['tM2'] = tM2
    df['avg'] = df[['tM1', 'tM2']].mean(axis=1)

    df.set_index(f'{SplitName}', inplace=True)

    #drop if any preview images are left in the dataframe before averaging
    df.drop(df[df.index.str.contains(r'Preview\d{3}', regex=True)].index, inplace=True)

    averages = df[['tM1', 'tM2', 'avg']].mean(axis=0).to_frame('avg').T

    std = df[['tM1', 'tM2', 'avg']].std(axis=0).to_frame('std').T

    df_new = pd.concat([df, averages, std])

    df_new.to_csv(f'{folder_path}/{SplitName}.csv')

    print(f"Saving df for {SplitName}")