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

# TODO:
# Open files
# for this no. of ND's calculate simulate 100 random more ND's and calc Manders with this and LT channel
# find distribution of Manders coeff from random ND distribution
# (include nucleus exclusion and cell bounds for random ND distribution)
# check random distribution is gaussian to be able to perform the Welch test statistic on 

# choose folder with .lif files in 
# root = tk.Tk()
# root.withdraw()
# #open GUI to ask to open folder path and select folder to analyse
# folder_path = filedialog.askdirectory()
# extension = "*.lif" #folder extension of output from ImageJ Log files
# files = glob.glob(f"{folder_path}/{extension}", recursive=False)
#print(mM1)

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
    mM1 = []
    mM2 = []
    sM1_mean = []
    sM2_mean = []
    sM1_std = []
    sM2_std = []
    cnt = 0
    for file in tqdm(files, desc="Files", position=0):
        lif = LifFile(f"{file}")
        date = file.split('/')[-1].split('.')[0].split('_')[0]
        time = file.split('/')[-1].split('.')[0].split('_')[-1]
        
        for i in tqdm(range(len(lif.image_list)), desc="image list", position=1, leave=False):
            dims = lif.image_list[i]['dims']
            if dims[3] == 1 and dims[2] == 1: #check this is a single image not a t or z stack
                list_for_df_row = []
                list_for_df_row.append(date)
                list_for_df_row.append(time)
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
                n = 0
                for k in tqdm(range(num_simulations), desc="simulations", position=2,leave=False):
                    image_size = (512, 512)  # generate random image of same dimensions
                    image = np.zeros(image_size)
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

            # Baby I don't know how to finish your code, but I don't want to wake you up!
            # It looks really good. Like way better than anything we do in the lab!
            # You should be super pround of yourself!!! <3 <3



if __name__ == "__main__":
    app()