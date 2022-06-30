"""
#  EoR_research/project_driver.py

  Author : Hugo Baraer
  Supervision by : Prof. Adrian Liu
  Affiliation : Cosmic dawn group at McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules computing the require fields and parameters.
"""


#import classic python librairies
import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import emcee
import imageio
import corner
import powerbox as pbox
#import this project's modules
import z_re_field as zre
import Gaussian_testing as gauss
import FFT
import statistical_analysis as sa
import plot_params as pp
import statistics
from numpy import array
#import pyfftw
import zreion as zr
import z_reion_comparison as zrcomp



obj = zrcomp.input_info_field()
a = np.linspace(0,100,10)
obj.set_zreion(a,a,1.0,2.0,3.0)
storing_array = np.empty((10,10), dtype=object)

storing_array[0][:] = obj
np.save()
print(obj.zreioninfo.P_k_zre)
varying_input = 'F_star'
#varying_input = 'Turnover_Mass'
data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}
ionization_rates = []

storing_dict = {'21cmFAST':{'P_k_zre':[], 'ion_hist':[], 'P_k_dm':[], 'z_mean':[] ,'b_mz':[]}, 'z_reion_Hugo':{'P_k_zre':[], 'ion_hist':[], 'free_params':[]}}

z_reion_zre= np.load('zreion_for_Hugo.npz')['zreion']
reion_hist_zreion = pp.reionization_history(np.linspace(5, 15, 100), z_reion_zre, plot = False)

# z_re_box= np.load('zre.npy')
# density_field = np.load('density.npy')
# overzre, zre_mean = zre.over_zre_field(z_re_box)
#
# reion_hist_zreion_0593 = pp.reionization_history(np.linspace(5, 15, 100), zre_zreion, plot=False)
#pp.plot_multiple_ionhist(ion_rates, dictt,varying_input,zreion=reion_hist_zreion, zreion2=reion_hist_zreion_0593)
#pp.plot_variational_range_1dict(dictt, varying_name=varying_input)

bmzs = []

Heff_range = np.linspace(20, 40,10)
T_vir_range = np.linspace(3.5,8.5,10)
Heff_ver, T_vir_hor = np.meshgrid(Heff_range, T_vir_range)


# zre_mean = [6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
# random_seed = [12345,54321,23451,34512,45123]
#random_seed = random_seed[4]
for varying_in_value in tqdm(Heff_range):
    #adjustable parameters to look out before running the driver
    #change the dimension of the box and see effect

    use_cache = False # uncomment this line to re-use field and work only on MCMC part
    box_dim = 143 #the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    radius_thick = 2. #the radii thickness (will affect the number of bins and therefore points)
    box_len = 143 #int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM":box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
    astro_params = p21c.AstroParams({ "NU_X_THRESH":500, "HII_EFF_FACTOR": 46, "ION_Tvir_MIN":4.69897 }) #"HII_EFF_FACTOR":Heff, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
    flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": False})
    #add astro_params
    compare_with_james = False
    #p21c.global_params.FIND_BUBBLE_ALGORITHM = 1
    initial_conditions = p21c.initial_conditions(
    user_params = user_params,
    cosmo_params = cosmo_params
    #random_seed = random_seed
    )
    dummy_count = 0
    if compare_with_james:
        dummy_count +=1
        continue

    if os.path.exists('b_mz.npy') and os.path.exists('bmz_errors.npy') and os.path.exists('kvalues.npy') and use_cache and os.path.exists('zre.npy'):
        #bmz_errors = np.load('bmz_errors.npy')
        #b_mz = np.load('b_mz.npy')
        kvalues = np.load('kvalues.npy')
        z_re_box= np.load('zre.npy')
        density_field = np.load('density.npy')
        overzre, zre_mean = zre.over_zre_field(z_re_box)

    else :
        redshifts = np.linspace(5,15,50)
        b_mz, kvalues, data_dict, density_field, cmFAST_hist, zre_PP = sa.generate_bias(redshifts, initial_conditions, box_dim, astro_params, flag_options, varying_input,
                          varying_in_value, data_dict=data_dict,comp_zre_PP=True)

        zre_zreion = zr.apply_zreion(density_field, data_dict[zre_mean], data_dict["a50"],data_dict["b50"],data_dict["k50"])
        zreion_zre_PP = pp.ps_ion_map(zre_zreion)
        zreion_hist = pp.reionization_history(zre_zreion)


        bmzs.append(b_mz.tolist())
        #print(ionization_rates)
        print(data_dict)

    #y_plot_fit = sa.lin_bias(kvalues, 0.564,0.593,0.185)

print("bmzs = ", bmzs)
print("dictt = ", data_dict)
print("ion_rates =" , ionization_rates)