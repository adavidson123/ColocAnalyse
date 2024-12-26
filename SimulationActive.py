import SimulationsFcns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image


#SimulationsFcns.change_size_blobs(blob_max_size=0.5, no_trials=50)

#SimulationsFcns.change_no_blobs(no_blobs=20, no_trials=50, max_no_blobs=100)

M1 = []
M2 = []
i = 0
image_size = (512, 512)  # Image dimensions as from sample images#

#initalize to zero two blank images to later perform colocalization on
image_LT = np.zeros(image_size)
image_ND = np.zeros(image_size)

num_simulations = 1

while i < num_simulations:

    image_LT_blobs = SimulationsFcns.add_blobs_to_image(image_LT, 106, 0.0013, 0.00054)
    image_ND_blobs = SimulationsFcns.add_blobs_to_image(image_ND, 40, 0.0013, 0.00054)
    m1, m2 = SimulationsFcns.calc_manders(image_LT_blobs, image_ND_blobs)
    M1.append(m1)
    M2.append(m2)

    if i == int(num_simulations / 4):
        print(r'25% done!')
    if i == int(num_simulations / 2):
        print(r'50% done!')
    if i == int((num_simulations / 4) * 3):
        print(r'75% done!')

    i = i+1

#print(M1)
#print(M2)

#######################################################

# binary_image_LT = np.uint8(image_LT_blobs * 255)
# normalized_LT = binary_image_LT / 255.0  # This normalizes it to [0, 1]
# print(normalized_LT[0])
# binary_image_ND = np.uint8(image_ND_blobs * 255)
# normalized_ND = binary_image_ND / 255.0  # This normalizes it to [0, 1]
# colored_image_LT = plt.cm.Greens(normalized_LT)
# colored_image_ND = plt.cm.Reds(normalized_ND)
# print(colored_image_LT[0])
# image = Image.fromarray(colored_image_LT)
#colored_image_rgb_LT = (colored_image_LT[:, :, :3] * 255).astype(np.uint8)
#colored_image_rgb_ND = (colored_image_ND[:, :, :3] * 255).astype(np.uint8)

#combined_image = np.maximum(colored_image_rgb_LT, colored_image_rgb_ND)

# Convert combined numpy array back to PIL image
#
# final_image = Image.fromarray(combined_image)
#final_image.show()

# show last image generated with channels overlapping and save to folder as an example image
#######################################################

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].hist(np.array(M1))
ax[1].hist(np.array(M2))
ax[0].set_title('M1')
ax[1].set_title('M2')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Manders Coefficients')
plt.show()

M1_mean = np.mean(M1)
M1_std = np.std(M1)
M2_mean = np.mean(M2)
M2_std = np.std(M2)

print(f"M1 mean: {M1_mean}")
print(f"M1 std: {M1_std}")
print(f"M2 mean: {M2_mean}")
print(f"M2 std: {M2_std}")

## do I need to confirm this distribution is Gaussian with chi-square test??

# save simulation data into a df for accessing for WelchTest.py

n_M1 = len(M1)
n_M2 = len(M2)

M1_err = M1_mean / np.sqrt(n_M1)
M2_err = M2_mean / np.sqrt(n_M2)

df = pd.DataFrame()

df['M1'] = [M1_mean]
df['M1_std'] = [M1_std]
df['M1_err'] = [M1_err]
df['M2'] = [M2_mean]
df['M2_std'] = [M2_std]
df['M2_err'] = [M2_err]

print(df)

## make new folder to save these files to
new_path = f'C:/Users/pa112h/Documents/PhD_1/LysoTracker/Simulations/{num_simulations}_sims_newsize'

if not os.path.exists(new_path):
    os.mkdir(new_path)
    print(f"Folder '{new_path}' created.")
else:
    print(f"Folder '{new_path}' already exists.")

#save these plots to the same folder
fig.savefig(f"{new_path}/Histogram.png")

#save average_df dataframe to csv file in case wanted for future plotting
df.to_csv(f'{new_path}/MandersCoeffs.csv')

##########################################
# Round 1 (500):
# M1 mean: 0.03036014734210514
# M1 std: 0.011428182673736083
# M2 mean: 0.07619758125960982
# M2 std: 0.07619758125960982
########################################
# Round 2 (500):
# M1 mean: 0.029713580025211683
# M1 std: 0.010777876178010829
# M2 mean: 0.07455276302306266
# M2 std: 0.027178915594240346
########################################