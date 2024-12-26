# find average number of pixels of lysotracker and NDs on average in the channels in imagej

import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
from readlif.reader import LifFile
from pims import Bioformats
import trackpy as tp
import statistics as st


new = LifFile('C:/Users/pa112h/Documents/PhD_1/LysoTracker/241120_confocal/241120_1h.lif')

LT = []
ND = []

for i, _ in enumerate(new.image_list):

    # Access a specific image directly
    img = new.get_image(i)

    pil_image_LT = img.get_frame(c=0)
    pil_image_ND = img.get_frame(c=1)
    #type(pil_image)

    np_image_LT = np.array(pil_image_LT)
    np_image_ND = np.array(pil_image_ND)

    f_LT = tp.locate(np_image_LT, 15, invert=False, minmass=400, separation=5)
    f_ND = tp.locate(np_image_ND, 17, invert=False, minmass=400, separation=5)

    #tp.annotate(f_LT, np_image_LT)
    #tp.annotate(f_ND, np_image_ND)
    #plt.show()

    #print(f"No. Lysosomes: {f_LT.shape[0]}")
    #print(f"No. Nanodiamonds: {f_ND.shape[0]}")

    LT.append(f_LT.shape[0])
    ND.append(f_ND.shape[0])

print(f"Average LT: {st.mean(LT)}")
print(f"Average ND: {st.mean(ND)}")

##################
#Average LT: 106.2
#Average ND: 39.6
##################