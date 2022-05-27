"""
  EoR_research/project_driver.py

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


varying_input = 'Turnover_Mass'
data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}
ionization_rates = []

z_reion_zre= np.load('zreion_for_Hugo.npz')['zreion']
reion_hist_zreion = pp.reionization_history(np.linspace(5, 12, 100), z_reion_zre, plot = False)


Heff_range = np.linspace(9,10,5)
for Heff in tqdm(Heff_range):
    #adjustable parameters to look out before running the driver
    #change the dimension of the box and see effect

    use_cache = False # uncomment this line to re-use field and work only on MCMC part
    box_dim = 143 #the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    radius_thick = 2. #the radii thickness (will affect the number of bins and therefore points)
    box_len = 143 #int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM":box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
    astro_params = p21c.AstroParams({ "M_TURN" : Heff }) #"HII_EFF_FACTOR":Heff, "M_TURN" : Heff
    flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": True})
    #add astro_params

    initial_conditions = p21c.initial_conditions(
    user_params = user_params,
    cosmo_params = cosmo_params,
    )



    if os.path.exists('b_mz.npy') and os.path.exists('bmz_errors.npy') and os.path.exists('kvalues.npy') and use_cache and os.path.exists('zre.npy'):
        #bmz_errors = np.load('bmz_errors.npy')
        #b_mz = np.load('b_mz.npy')
        kvalues = np.load('kvalues.npy')
        z_re_box= np.load('zre.npy')
        density_field = np.load('density.npy')

    else :




        """Compute the over-redshfit or reionization and overdensity"""
        #Compute the reionization redshift from the module z_re
        #5,7.4,7.45,7.5,7.55,7.6,7.65,7.7,7.75,7.8,7.85, 7.9,7.95,8.0,8.05,8.1,8.15,8.2,8.25,8.3,8.35,8.4,8.45,8.5,8.55,8.6,8.65,8.7,8.75,8.8,8.85,8.9,8.95,9.0,9.05,9.1,9.15,9.2,9.25,9.3,9.35,9.4,9.45,9.5,9.55,9.6,9.65,9.7,9.75,9.8,9.85,9.9,9.95,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,12.2,12.4,12.6,12.8,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.5,16.5,17,17.5,18,18.5,19,19.5,20,20.5]
        zre_range = np.linspace(5,15,50)
        z_re_box = zre.generate_zre_field(zre_range, initial_conditions,box_dim, astro_params, flag_options,comP_ionization_rate=False)
        overzre, zre_mean = zre.over_zre_field(z_re_box)

        #np.save('./zre',z_re_box)
        #np.save('./density', perturbed_field.density)
        #np.save('./b_mz', b_mz)
        #np.save('./bmz_errors', bmz_errors)
        #np.save('./kvalues', kvalues)


    """This section uses the redshfit of reionization field to evaluate the power spectrum with the premade pacakge"""


    redshifts = np.linspace(5, 12, 100)





    cmFast_hist, width_50_21, width_90_21 = pp.reionization_history(redshifts,z_re_box, comp_width=True)
    ionization_rates.append(cmFast_hist)
    nb_bins = 20
    nb_bins = int(nb_bins)

    #pp.plot_multiple_ionhist([cmFast_hist, reion_hist_zreion],['21cmFAST default input','z-reion'])

    #pp.ionization_movie(np.linspace(5,15,20), z_re, 143, 'test_movie.gif')
    perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    #p_k_density, kbins_density = pbox.get_power(density_field, 100)
    p_k_zre, kbins_zre = pbox.get_power(overzre, 100)
    #b_mz = np.sqrt(np.divide(p_k_zre, p_k_density))


    zre_pp = pp.ps_ion_map(overzre, nb_bins, 143, logbins=True)
    den_pp = pp.ps_ion_map(density_field, nb_bins, 143, logbins=True)

    delta = 0.1
    Xd, Yd, overd_fft, freqd = FFT.compute_fft(density_field, delta, box_dim)
    Xd, Yd, overzre_fft, freqd = FFT.compute_fft(overzre, delta, box_dim)
    cross_matrix = overd_fft.conj() * overzre_fft
    #cross_matrix =  overzre_fft.conj().T *overd_fft
    b_mz1 = np.sqrt(np.divide(zre_pp, den_pp))
    #b_mz2 = np.sqrt(np.divide(p_k_zre, p_k_density))

    values_cross, count_cross = sa.average_overk(143,cross_matrix,nb_bins, logbins=True)
    cross_pp = np.divide(values_cross, count_cross)
    # k_values = np.linspace(kbins_zre.min(), kbins_zre.max(), len(cross_pp))
    k_values = np.linspace(kbins_zre.min(), kbins_zre.max(), nb_bins+1)
    k_values = np.logspace(np.log10(kbins_zre.min()), np.log10(kbins_zre.max()), nb_bins+1)


    cross_cor = np.divide(cross_pp/8550986.582903482,np.sqrt(((zre_pp/8550986.582903482)*(den_pp/8550986.582903482))))

    """these lines plots the linear bias as a function of the kvalues"""




    '''
    MCMC analysis and posterior distribution on the b_mz 
    '''

    #errs = np.ones_like(b_mz)*0.05

    #no b_mz fitting
    b_mz = b_mz1[1:]
    kbins_zre = k_values[1:]
    cross_cor = cross_cor[1:]

    data_dict = sa.run_MCMC_free_params(kbins_zre,b_mz,cross_cor, data_dict=data_dict, varying_input=varying_input, varying_in_value= Heff)


    data_dict['width50'].append(width_50_21)
    data_dict['width90'].append(width_90_21)

    print(ionization_rates)
    print(data_dict)
    x0 = np.linspace(0.06, 10, 1000)
    #y_plot_fit = sa.lin_bias(kvalues, 0.564,0.593,0.185)




print(data_dict)
print(ionization_rates)