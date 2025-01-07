from readlif.reader import LifFile
import os
import matplotlib
from matplotlib import pyplot as plt

file_path = os.path.join('c:/Users/pa112h/Documents/PhD_1/LysoTracker/241120_confocal/', '241120_1h.lif')

lif = LifFile(f"{file_path}")

image = lif.get_image(0)
c0 = image.get_frame(c=2)

plt.imshow(c0, cmap='gray')
plt.show()