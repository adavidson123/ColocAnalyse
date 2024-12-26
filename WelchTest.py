# perform a Welch test (a t-test for two distrbutions with different variances)

from scipy import stats
import numpy as np
import pandas as pd

# load in data from averages df as an array 

# load in data from simulations 

#stats.ttest_ind()

def welch_test(x1_bar, x2_bar, s1_bar, s2_bar):
    num = x1_bar - x2_bar
    denom = np.sqrt(s1_bar **2 + s2_bar ** 2)
    t = num / denom
    return t

def degrees_of_freedom(s1_bar, s2_bar, std1, std2):
    N1 = (std1 / s1_bar) ** 2
    N2 = (std2 / s2_bar) ** 2
    num = (((std1 ** 2) / N1) + ((std2 ** 2) / N2)) ** 2
    denom = ((std1 ** 4) / ((N1 ** 2) * (N1 - 1))) + ((std2 ** 4) / ((N2 ** 2) * (N2 - 1)))
    nu = num / denom
    return nu

exp_data = 'C:/Users/pa112h/Documents/PhD_1/LysoTracker/AllDataPlots/AveragedData.csv'
sim_data = 'C:/Users/pa112h/Documents/PhD_1/LysoTracker/Simulations/500_sims/MandersCoeffs.csv'

df_exp_data = pd.read_csv(exp_data, index_col=0)
df_sim_data = pd.read_csv(sim_data)

#print(df_exp_data)
#print(df_sim_data)

# load in simulated errors and means 
M1_sim = df_sim_data.loc[0]['M1']
M1_sim_err = df_sim_data.loc[0]['M1_err']
M1_sim_std = df_sim_data.loc[0]['M1_std']
M2_sim = df_sim_data.loc[0]['M2']
M2_sim_err = df_sim_data.loc[0]['M2_err']
M2_sim_std = df_sim_data.loc[0]['M2_std']

# load in lists of experimental errors and means
#M1_exp = df_exp_data['tM1']
#M1_exp_err = df_exp_data['tM1_err']
#M2_exp = df_exp_data['tM2']
#M2_exp_err = df_exp_data['tM2_err']

# iterate through and perform a Welch test

M1_Welch_stat = []
M1_nu_val = []
for i, value in enumerate(df_exp_data['tM1']):
    #perform the Welch test
    index = df_exp_data.index[i]
    exp_err = df_exp_data.loc[index]['tM1_err']
    M1_Welch = welch_test(M1_sim, value, M1_sim_err, exp_err)
    #print(M1_Welch)
    M1_Welch_stat.append(abs(M1_Welch))
    #calculate the no. of degrees of freedom
    exp_std = df_exp_data.loc[index]['tM1_std']
    M1_nu = degrees_of_freedom(M1_sim_err, exp_err, M1_sim_std, exp_std)
    M1_nu_val.append(int(M1_nu))


M2_Welch_stat = []
M2_nu_val = []
for i, value in enumerate(df_exp_data['tM2']):
    index = df_exp_data.index[i]
    exp_err = df_exp_data.loc[index]['tM2_err']
    M2_Welch = welch_test(M2_sim, value, M2_sim_err, exp_err)
    #print(M1_Welch)
    M2_Welch_stat.append(abs(M2_Welch))
    exp_std = df_exp_data.loc[index]['tM2_std']
    M2_nu = degrees_of_freedom(M2_sim_err, exp_err, M2_sim_std, exp_std)
    M2_nu_val.append(int(M2_nu))

print(f"M1 Welch stats list: {M1_Welch_stat}")
print(f"M2 Welch stats list: {M2_Welch_stat}")
print(f"M1 nu values list: {M1_nu_val}")
print(f"M2 nu values list: {M2_nu_val}")
