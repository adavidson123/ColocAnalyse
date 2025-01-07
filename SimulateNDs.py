# import tkinter as tk
# from tkinter import filedialog
from readlif.reader import LifFile
import glob
import SimulationsFcns
import numpy as np
import trackpy as tp
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import WelchTest
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt

# TODO:
# Open files
# (include nucleus exclusion and cell bounds for random ND distribution)


import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config_sim_nds")
def app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    files = []
    for file in os.listdir(cfg.dir_path):
        if file.endswith(".lif"):
            files.append(os.path.join(cfg.dir_path, file))


    num_simulations = cfg.num_simulations


    df = pd.DataFrame(columns=['Date','Time','mM1','mM2','sM1','sM2','sM1_std','sM2_std'])

    cnt = 0
    for file in tqdm(files, desc="Files", position=0):
        #################################################################
        # find date and time after removal from incubation
        #################################################################
        lif = LifFile(f"{file}")
        head, tail = os.path.split(file)
        date = tail.split('_')[0]
        time = tail.split('_')[-1].replace('h.lif', '')
        
        for i in tqdm(range(len(lif.image_list)), desc="image list", position=1, leave=False):
            dims = lif.image_list[i]['dims']
            if dims[3] == 1 and dims[2] == 1: #check this is a single image not a t or z stack
                list_for_df_row = []
                list_for_df_row.append(date)
                list_for_df_row.append(float(time))
                ##############################################################
                # calc manders coeff for LT and ND images
                #############################################################
                img = lif.get_image(i) # get each image in .lif file
                pil_image_LT = img.get_frame(c=0) # get LT channel image
                pil_image_ND = img.get_frame(c=1) # get ND channel image
                np_image_LT = np.array(pil_image_LT) # convert LT channel to array
                np_image_ND = np.array(pil_image_ND) # convert ND image to array
                m1, m2 = SimulationsFcns.calc_manders(np_image_LT, np_image_ND)
                #mM1.append(m1)
                list_for_df_row.append(m1)
                #mM2.append(m2)
                list_for_df_row.append(m2)
                #####################################################################
                # measure no. of NDs in each image and generate similar random images
                # calc manders coeffs of these and save average and std 
                #####################################################################
                f_ND = tp.locate(np_image_ND, 17, invert=False, minmass=400, separation=5)
                ND_no = f_ND.shape[0]
                sM1 = []
                sM2 = []

                for k in tqdm(range(num_simulations), desc="simulations", position=2,leave=False):
                    image_size = (512, 512)  # generate random image of same dimensions
                    image = np.zeros(image_size)
                    # size of blobs generated here is from previous estimates
                    # improve this so calculate average size of blobs in each image to be used for 
                    # random generation comparison???
                    image_ND_blobs = SimulationsFcns.add_blobs_to_image(image, ND_no, 0.0013, 0.00054)
                    sm1, sm2 = SimulationsFcns.calc_manders(np_image_LT, image_ND_blobs)
                    sM1.append(sm1)
                    sM2.append(sm2)
                #sM1_mean.append(np.mean(sM1))
                list_for_df_row.append(np.mean(sM1))
                #sM2_mean.append(np.mean(sM2))
                list_for_df_row.append(np.mean(sM2))
                #sM1_std.append(np.std(sM1))
                list_for_df_row.append(np.std(sM1))
                #sM2_std.append(np.std(sM2))
                list_for_df_row.append(np.std(sM2))
                ########################################################################
                # create a new df with format date|time|mM1|mM2|sM1|sM2|sM1_std|sM2_std
                #  save each value into the df accordingly
                ########################################################################
                df.loc[cnt] = list_for_df_row
                cnt+=1

    # Save the dataframe to the same folder as cfg.dir_path
    output_file_path = os.path.join(cfg.dir_path, 'Manders_df.csv')
    df.to_csv(output_file_path, index=False)

    
    ########################################################################
    # Mean, std and sem on rows with the same Time values 
    # For same Times, check if distribution is Gaussian using Shapiro-Wilks test
    # Calculate t-test indep with unequal variance (Welch Test) and add p-value to df
    # Calculate Wilcoxon rank-sum test and add p-value to df
    ########################################################################
    results_df = pd.DataFrame(columns=['Time', 'mM1_avg', 'mM2_avg', 'sM1_avg', 'sM2_avg', 
                                       'mM1_std', 'mM2_std', 'sM1_std', 'sM2_std',
                                       'mM1_sem', 'mM2_sem', 'sM1_sem', 'sM2_sem',
                                       'mM1_normal', 'mM2_normal', 'sM1_normal', 'sM2_normal',
                                       #'M1_tstat', 
                                       #'M1_tp', 
                                       #'M2_tstat', 
                                       #'M2_tp', 
                                       'M1_rankp', 'M2_rankp'])

    cnt_2 = 0
    # Check if M1 and M2 columns are normally distributed for the same Time values
    unique_times = df['Time'].unique()
    for time in unique_times:
        subset = df[df['Time'] == time]
        mM1_values = subset['mM1']
        mM2_values = subset['mM2']
        sM1_values = subset['sM1']
        sM2_values = subset['sM2']
        
        # Shapiro-Wilk test for normality
        stat_mM1, p_mM1 = stats.shapiro(mM1_values)
        stat_mM2, p_mM2 = stats.shapiro(mM2_values)
        stat_sM1, p_sM1 = stats.shapiro(sM1_values)
        stat_sM2, p_sM2 = stats.shapiro(sM2_values)
        
        # Just return values and later can set significane level and check if gaussian or not
        #mM1_normal = p_mM1 > 0.05
        #mM2_normal = p_mM2 > 0.05
        #sM1_normal = p_sM1 > 0.05
        #sM2_normal = p_sM2 > 0.05
        
        # Calculate averages
        mM1_avg = mM1_values.mean()
        mM2_avg = mM2_values.mean()
        sM1_avg = sM1_values.mean()
        sM2_avg = sM2_values.mean()

        # Calculate standard deviations
        mM1_std = mM1_values.std()
        mM2_std = mM2_values.std()
        sM1_std = sM1_values.std()
        sM2_std = sM2_values.std()

        # Calculate standard errors 
        mM1_sem = mM1_values.sem()
        mM2_sem = mM2_values.sem()
        sM1_sem = sM1_values.sem()
        sM2_sem = sM2_values.sem()

        # Calculate ttest statistic
        #M1_tstat = stats.ttest_ind(mM1_values, sM1_values, equal_var = False)
        #M2_tstat = stats.ttest_ind(mM2_values, sM2_values, equal_var = False)

        # Calculate Wilcoxon rank-sum statistic
        M1_ranksum = stats.ranksums(mM1_values, sM1_values, alternative='greater') 
        # use alternative='greater' as want to determine if mM1 the measured values are greater than the sM1 simulated values
        M2_ranksum = stats.ranksums(mM2_values, sM2_values, alternative='greater')

        # List with values for a dataframe
        row_for_results = [time, mM1_avg, mM2_avg, sM1_avg, sM2_avg, mM1_std, mM2_std, sM1_std, sM2_std, 
                           mM1_sem, mM2_sem, sM1_sem, sM2_sem, 
                           #mM1_normal, mM2_normal, sM1_normal, sM2_normal,
                           p_mM1, p_mM2, p_sM1, p_sM2,
                           #M1_tstat.statistic,
                            #M1_tstat.pvalue, 
                            #M2_tstat.statistic, 
                            #M2_tstat.pvalue,
                           M1_ranksum.pvalue, M2_ranksum.pvalue]
        
        # 
        results_df.loc[cnt_2] = row_for_results
        cnt_2+=1
    
    
    # Order the rows by the Time column...
    results_df.sort_values(by='Time', inplace=True)

    # Save the results dataframe to the same folder as cfg.dir_path
    output_results_path = os.path.join(cfg.dir_path, 'Manders_average_df.csv')
    results_df.to_csv(output_results_path, index=False)

    #####################################################################################
    # What plotting functionality do I want to add from the dataframes...??
    # Add plotting in from TryPlotting.py
    # Save this plot as a png file in the same folder as the results
    # Add plotting for different dates separately...??
    #####################################################################################
    
    x_ticks = range(len(results_df['Time']))

    plt.figure(figsize=(7, 5))

    plt.errorbar(x_ticks, results_df['sM2_avg'], yerr=results_df['sM2_std'], linestyle='', marker='_', capsize=5, 
                color='#66c2a4', label='Simulated')
    plt.errorbar(x_ticks, results_df['mM2_avg'], yerr=results_df['mM2_std'], linestyle='', marker='_', capsize=5, 
                color='#006d2c', label='Measured')

    plt.xlabel('Time (hours)')
    plt.ylabel('Manders Coefficient')
    plt.title('Overlap of NDs with Lysosomes')
    plt.xticks(ticks=x_ticks, labels=[str(tick) for tick in results_df['Time']])
    plt.legend()
    
    plt.savefig(os.path.join(cfg.dir_path, 'Manders_plot.png'))



if __name__ == "__main__":
    app()