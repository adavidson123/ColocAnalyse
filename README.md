My file formats and naming system:
- files in folders with the date and _confocal e.g. yymmdd_confocal (241122_confocal)
- files from Lecia are saved as .lif files which contain series of Images 
    - z-stacks and timelapses were saved as different experiments


CoLysoLisation.ijm
- macro to run in ImageJ
- need to select the file you want to analyse 
  - select "Hyperstack" as format to open image
  - manually select the images you want to open and click OK
- run code and will split each image into channels and select C1, C2 of split channels to perform ImageJ Coloc2 function on
  - need images to be saved in a format where C1 and C2 will be the LysoTracker dye and the NDs
  - only tested with individual images although Coloc2 should work with z-stacks so think they should also work in this case (Coloc2 does not work with timelapses without splitting time channels first)
- output of code saved to Log in ImageJ which is restarted intially when running the code so should only contain information about that experiment in it
  - currently need to manually save the Log file and have been saving as yymmdd_xh_Log.txt
  - will try to ammend this so it manually saves but that function isn't working currently and this isn't essential for this small number of experiments
  

Coloc2Analyse.py
- when window opens select the folder that the ***_Log.txt files are saved in from the output of CoLysoLisation.ijm
- code will process each .txt file in the folder and select out the image name, tM1, tM2 and an average of these to add to a df which is then saved in the same folder as a .csv file with the same name as the original .lif file
- will try to process all .txt files in the folder and expecting the same format input as the ImageJ output so remove any other formatted .txt files from the folder before running
- files not labelled as an 'Image' will be removed from the dataframe


ColocPlot.py
- takes input folders of dataframes with each date attached from Coloc2Analyse
- outputs plots e.g. average coloc over time plots w/ SD error bars including for tM1 and tM2
- saves plots to same folder as data analysed from
- want to implement: saves output dataframe with average values ready for plotting of all experiments


ColocCompare.py
- plots the results of all experiements onto one complete plot 
- searches through all folders in sub directories to find the average value .csv files from each date folder and accumulates this into one plot to compare different trials
Issues:
- assumes each dataset has data avilable from each timestamp for the experiment (doesn't cope well with missing data) - try to ammend this when I have time


NewSimulations.py
- to play around with before proper functions for carrying out the simulations were made


SimulationFcns.py
- randomly simulates gaussian spheres and places them randomly on a 256 x 256 pixel image
- then calculates theoretical Manders coefficients from simulated images
- will contain all master functions for performing different simulations


SimulationActive.py
- import in SimulationFcns
- carry out simulations in an easier to see format
- save average, std and std errors into a .csv file to be used in WelchTest.py


CountNDs.py
- will count the number of lysosomes and NDs on average in all the .lif files in a folder
- these values can then be used for simulations of randomly distributed particles to assess if the experimental results are significantly different from what you could sxpect from a random distribution
- currently this can estimate numebr from using trackpy:
  - this works when specifying input values however depending on the brightness of individual images as well as particualr zooms etc that were used for each image it may not translate well to iterating through many sample images
  - can try a few iterations with different images and see if it can be optimized? 
  - may need to use a method with different thresholding?


WelchTest.py
- performs a Welch test on simulated data compared to data taken from experiments


SimulateNDs.py
- load in images from .lif files
- use the LT image as a base and comapre random distribution of same number of NDs
- for any .lif file that has channel 1 as lysosome/ mitochondria and channel 2 as the ND channel, will use the base lyso/ mito channel and generate similar numbers of ND at random points on the screen, calculate Manders coeffs for each case 