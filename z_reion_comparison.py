"""
  EoR_research/tested_things.py

  Author : Hugo Baraer
  Supervision by : Prof. Adrian Liu
  Affiliation : Cosmic dawn group at McGill University
  Date of creation : 2022-05-20

  This module contains chunk of commented tested code to clear up and organize the driver
"""

import py21cmfast as p21c
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistical_analysis as sa
import plot_params as pp
import zreion as zr
import imageio

def compare_reion_hist(zremean_Hugo, alpha_hugo, b_0_Hugo, k_0_Hugo, ion_rates, james_zre_means, james_alphas, james_k_0, plot = True, saveforgif = False, filenames = [], imnb = 0, title = ''):
    '''
    This function compares the reionization histories from 21cmFAST and the fitted parameters from Hugo and James's method.
    :param zremean_Hugo:  the mean redshfit of reionization computed with Hugo's method
    :param alpha_hugo:  the alpha computed with Hugo,s method
    :param b_0_Hugo:  the b_0 parameter computed by hugo (very very often 1)
    :param k_0_Hugo:  the k_0 parameter computed by hugo
    :param ion_rates:  the ionization history computed with 21cmFAST reionization data
    :param james_zre_means:  the mean redshfit of reionization computed with James's method
    :param james_alphas: the alpha computed with Hugo's method
    :param james_k_0:
    :param plot = True: plots reion history if True
    :return:
    '''
    box_dim = 143  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    box_len = 143  # int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
    initial_conditions = p21c.initial_conditions(user_params=user_params, cosmo_params=cosmo_params, )
    perturbed_field = p21c.perturb_field(redshift=zremean_Hugo, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_me = zr.apply_zreion(density_field, zremean_Hugo, alpha_hugo, k_0_Hugo, 100, b0=b_0_Hugo)
    perturbed_field = p21c.perturb_field(redshift=james_zre_means, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_james = zr.apply_zreion(density_field, james_zre_means, james_alphas, james_k_0, 100)
    reion_hist_zreion_me = pp.reionization_history(np.linspace(5, 15, 100), zre_zreion_me, plot=False)
    reion_hist_zreion_James = pp.reionization_history(np.linspace(5, 15, 100), zre_zreion_james, plot=False)

    #pp.plot_21zreion_ionhist([reion_hist_zreion_me, ion_rates, reion_hist_zreion_James], saveforgif=saveforgif)

    return pp.plot_21zreion_ionhist([reion_hist_zreion_me, ion_rates, reion_hist_zreion_James], saveforgif=True, filenames=filenames, imnb=imnb, title = title)

def plot_variational_range_James(dict1, james_alpha, james_k_0, varying_name='Heff', varying_title = 'Heff'):
    '''
    This function generates the plot of the variational range for 3 parameters with 1 different input
    :param dict1: the dictionnary of the first input
    :type dict1: dict
    :return:
    :rtype:
    '''
    a = dict1
    fig3, ax3 = plt.subplots(2, 1, sharex='col', sharey='row')
    cont300 = ax3[0].scatter(a['{}'.format(varying_name)], james_alpha, color='r')
    ax3[1].scatter(a['{}'.format(varying_name)], james_k_0, color='r')


    plt.setp(ax3[1], xlabel='{}'.format(varying_title))
    plt.setp(ax3[0], ylabel=r'$\alpha$')
    plt.setp(ax3[1], ylabel=r'$k_0$')

    plt.show()