import numpy as np
from PIL import Image
import random
from skimage import measure, filters
import matplotlib.pyplot as plt
import pandas as pd
import os

#make this into one big simulations function to use in a different python file?


#generates a gaussian shaped blob at a specified with its center at center and of size
def generate_gaussian_blob(image_size, center, size):
    # Create a 2D grid of the image size
    x, y = np.meshgrid(np.linspace(0, 1, image_size[0]), np.linspace(0, 1, image_size[1]))
    
    # Create a 2D Gaussian distribution with given center and size
    d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    blob = np.exp(-(d**2 / (2.0 * size**2)))
    
    return blob

#function to generate blobs and add them to the input image
def add_blobs_to_image(image, num_blobs, blob_size_mean, blob_size_variance):

    for _ in range(num_blobs):
        # Randomly choose center position for the blob
        center = (random.uniform(0, 1), random.uniform(0, 1))
        
        # Randomly choose size for the blob (Gaussian standard deviation)
        blob_size = random.gauss(blob_size_mean, blob_size_variance)
        
        # Generate the Gaussian blob
        blob = generate_gaussian_blob(image_size, center, blob_size)
        
        # Add the blob to the image, making sure not to exceed max intensity
        image = np.maximum(image, blob)

    return image
    
# function to threshold images and convert them to binary images
def calc_manders(image1, image2):

    threshold1 = filters.threshold_otsu(image1)
    threshold2 = filters.threshold_otsu(image2)

    binary_image1 = image1 > threshold1
    binary_image2 = image2 > threshold2

    m1 = measure.manders_coloc_coeff(binary_image1, binary_image2)
    m2 = measure.manders_coloc_coeff(binary_image2, binary_image1)

    return m1, m2

image_size = (512, 512)  # Image dimensions as from sample images#

#initalize to zero two blank images to later perform colocalization on
image1 = np.zeros(image_size)
image2 = np.zeros(image_size)

############################################################################################
# change no of blobs for a constant size
###########################################################################################

def change_no_blobs(no_blobs=10, 
                    blob_size_mean=0.001 # ~3 pixels in each direction
                    , blob_size_variance=0.0002, no_trials=10, max_no_blobs=10):
    
    #add blobs to one image
    blobs_image1 = add_blobs_to_image(image1, no_blobs, blob_size_mean, blob_size_variance)

    #make df for results of different trials to be added to
    df_number_variation_m1 = pd.DataFrame()
    df_number_variation_m2 = pd.DataFrame()

    #iterate through the number of trials in the simualtion to have multiple attempts to average over
    for trials in range(0, no_trials):
        if trials == int(no_trials/4):
            print(r"25% done!")
        if trials == int(no_trials/2):
            print(r"50% done!")
        if trials == int((no_trials/4) * 3):
            print(r"75% done!")
        i_number = []
        m1_number_variation = []
        m2_number_variation = []

        for i in range(1, max_no_blobs):

            i_number.append(i)
            image = add_blobs_to_image(image2, i, blob_size_mean, blob_size_variance)

            m1, m2 = calc_manders(image, blobs_image1)
            m1_number_variation.append(m1)
            m2_number_variation.append(m2)


        df_number_variation_m1[f'{trials}'] = m1_number_variation
        df_number_variation_m2[f'{trials}'] = m2_number_variation


    df_number_variation_m1.index = i_number
    df_number_variation_m2.index = i_number

    df_number_variation_m1_avg = pd.DataFrame(df_number_variation_m1.mean(numeric_only=True, axis=1))
    df_number_variation_m2_avg = pd.DataFrame(df_number_variation_m2.mean(numeric_only=True, axis=1))

    fig, ax = plt.subplots()
    ax.plot(df_number_variation_m1.index, df_number_variation_m1, 'k-', alpha=0.1) 
    ax.plot(df_number_variation_m1_avg.index, df_number_variation_m1_avg, 'k', label='M1')
    ax.plot(df_number_variation_m2.index, df_number_variation_m2, 'b-', alpha=0.1)
    ax.plot(df_number_variation_m2_avg.index, df_number_variation_m2_avg, 'b', label='M2') 
    ax.set(ylabel='Manders Coefficient',
        xlabel='Number of Particles')
    fig.legend()
    fig.tight_layout()

    df_number_variation_m1['std'] = df_number_variation_m1.std(numeric_only=True, axis=1)
    df_number_variation_m2['std'] = df_number_variation_m2.std(numeric_only=True, axis=1)

    df_number_variation_m1['std_err'] = df_number_variation_m1['std'] / (len(df_number_variation_m1.columns) - 1)
    df_number_variation_m2['std_err'] = df_number_variation_m2['std'] / (len(df_number_variation_m2.columns) - 1)

    df_number_variation_m1_avg['err'] = df_number_variation_m1['std_err']
    df_number_variation_m2_avg['err'] = df_number_variation_m2['std_err']

    #save average df into a folder to then be used to compare with actual results

    global_directory = "C:/Users/pa112h/Documents/PhD_1/LysoTracker/Simulations"

    ## make new folder to save these files to
    new_path = f'{global_directory}/NoVar_Image1_{no_blobs}_Image2_max_{max_no_blobs}'

    if not os.path.exists(new_path):
        os.mkdir(new_path)
        print(f"Folder '{new_path}' created.")
    else:
        print(f"Folder '{new_path}' already exists.")

    #save these plots to the same folder
    fig.savefig(f"{new_path}/No_Trials_{no_trials}.png")

    #save average_df dataframe to csv file in case wanted for future plotting
    df_number_variation_m1_avg.to_csv(f'{new_path}/M1_No_Trials_{no_trials}.csv')
    df_number_variation_m2_avg.to_csv(f'{new_path}/M2_No_Trials_{no_trials}.csv')

    print(f'Files saved in {new_path}, finished simulating!')



###################################################################################
#changing the size of the blob but keeping the number and variation the same
###################################################################################

def change_size_blobs(no_blobs1=10, no_blobs2=10,
                    blob_size_mean=0.001 # ~3 pixels in each direction
                    , blob_size_variance=0.0002, no_trials=10, blob_max_size=0.01):
    
    #add blobs to one image
    blobs_image1 = add_blobs_to_image(image1, no_blobs1, blob_size_mean, blob_size_variance)

    #make df for results of different trials to be added to
    df_size_variation_m1 = pd.DataFrame()
    df_size_variation_m2 = pd.DataFrame()

    #iterate through the size of trials in the simualtion to have multiple attempts to average over
    for trials in range(0, no_trials):
        if trials == int(no_trials/4):
            print(r"25% done!")
        if trials == int(no_trials/2):
            print(r"50% done!")
        if trials == int((no_trials/4) * 3):
            print(r"75% done!")

        i_number = []
        m1_size_variation = []
        m2_size_variation = []

        for i in np.arange(blob_size_mean, blob_max_size, 0.001):

            i_number.append(i)
            image = add_blobs_to_image(image2, no_blobs2, i, blob_size_variance)

            m1, m2 = calc_manders(image, blobs_image1)
            m1_size_variation.append(m1)
            m2_size_variation.append(m2)


        df_size_variation_m1[f'{trials}'] = m1_size_variation
        df_size_variation_m2[f'{trials}'] = m2_size_variation


    df_size_variation_m1.index = i_number
    df_size_variation_m2.index = i_number

    df_size_variation_m1_avg = pd.DataFrame(df_size_variation_m1.mean(numeric_only=True, axis=1))
    df_size_variation_m2_avg = pd.DataFrame(df_size_variation_m2.mean(numeric_only=True, axis=1))

    fig, ax = plt.subplots()
    ax.plot(df_size_variation_m1.index, df_size_variation_m1, 'k-', alpha=0.1) 
    ax.plot(df_size_variation_m1_avg.index, df_size_variation_m1_avg, 'k', label='M1')
    ax.plot(df_size_variation_m2.index, df_size_variation_m2, 'b-', alpha=0.1)
    ax.plot(df_size_variation_m2_avg.index, df_size_variation_m2_avg, 'b', label='M2') 
    ax.set(ylabel='Manders Coefficient',
        xlabel='Size of Particles')
    fig.legend()
    fig.tight_layout()

    df_size_variation_m1['std'] = df_size_variation_m1.std(numeric_only=True, axis=1)
    df_size_variation_m2['std'] = df_size_variation_m2.std(numeric_only=True, axis=1)

    df_size_variation_m1['std_err'] = df_size_variation_m1['std'] / (len(df_size_variation_m1.columns) - 1)
    df_size_variation_m2['std_err'] = df_size_variation_m2['std'] / (len(df_size_variation_m2.columns) - 1)

    df_size_variation_m1_avg['err'] = df_size_variation_m1['std_err']
    df_size_variation_m2_avg['err'] = df_size_variation_m2['std_err']

    #save average df into a folder to then be used to compare with actual results

    global_directory = "C:/Users/pa112h/Documents/PhD_1/LysoTracker/Simulations"

    ## make new folder to save these files to
    new_path = f'{global_directory}/SizeVar_Image1_{blob_size_mean}_Image2_max_{blob_max_size}'

    if not os.path.exists(new_path):
        os.mkdir(new_path)
        print(f"Folder '{new_path}' created.")
    else:
        print(f"Folder '{new_path}' already exists.")

    #save these plots to the same folder
    fig.savefig(f"{new_path}/No_Trials_{no_trials}.png")

    #save average_df dataframe to csv file in case wanted for future plotting
    df_size_variation_m1_avg.to_csv(f'{new_path}/M1_No_Trials_{no_trials}.csv')
    df_size_variation_m2_avg.to_csv(f'{new_path}/M2_No_Trials_{no_trials}.csv')

    print(f'Files saved in {new_path}, finished simulating!')

#changing the variation of blob size but keeping the number and size the same??

