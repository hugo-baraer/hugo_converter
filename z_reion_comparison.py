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


def compute_zre_zreion(alpha, b_0, k_0, zre_mean =8.011598437878444 ):
    '''
    This function computes the redshfit of reionization field from zreion for my  values given a set of parameters
    :param alpha: my value
    :param b_0: ""
    :param k_0:""
    :return: the redshfit of reionization field from zreion [3D array]
    '''

    #z_re = np.load('zre.npy')
    box_dim = 143  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    box_len = 143  # int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
    initial_conditions = p21c.initial_conditions(user_params=user_params, cosmo_params=cosmo_params)
    perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_me = zr.apply_zreion(density_field, 8.011598437878444, alpha, k_0, 100, b0=b_0)
    return zre_zreion_me


def compute_zre_james_me(alpha, b_0, k_0,james_alpha, james_k_0, james_mean):
    '''
    This function computes the redshfit of reionization field from zreion for my and James values given a set of parameters
    :param alpha: my value
    :param b_0: ""
    :param k_0:""
    :param james_alpha: James free parameter value
    :param james_k_0: ""
    :param james_mean: ""
    :return:
    '''
    alpha = 1.42
    b_0 = 1.00
    k_0 = 1.16
    james_alpha = 0.2400839581442675
    james_k_0 = 0.8343517851779568
    james_mean = 7.909691892213264
    #z_re = np.load('zre.npy')
    box_dim = 143  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    box_len = 143  # int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
    initial_conditions = p21c.initial_conditions(user_params=user_params, cosmo_params=cosmo_params)
    perturbed_field = p21c.perturb_field(redshift=8.011598437878444, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_me = zr.apply_zreion(density_field, 8.011598437878444, alpha, k_0, 100, b0=b_0)
    perturbed_field = p21c.perturb_field(redshift=james_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_james = zr.apply_zreion(density_field, james_mean, james_alpha, james_k_0, 100)
    return zre_zreion_me, zre_zreion_james

def compute_field_Adrian(zre_mean, initial_conditions, astro_params, flag_options, random_seed=12345):
    '''
    This functions computes the desired field for Adrian (density field, xH field and brighness temperature field and save them as npz files)
    :param zre_mean: [float] the mean redshfit of reionization computed at
    :param initial_conditions: [obj] the 21cmFAST initial conditions objects
    :param astro_params: [obj] the astro input parameter of 21cmFASt
    :param flag_options: [obj] the flag options input parameter of 21cmFASt
    :param random_seed: [int] the random seed at wich to compute the field (default is 12345)
    :return:
    '''
    perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    #print(density_field == p21c.run_coeval(redshift = zre_mean, user_params = user_params,cosmo_params = cosmo_params, astro_params = astro_params).density)
    ionized_box = p21c.ionize_box(redshift=zre_mean, init_boxes=initial_conditions, astro_params=astro_params,
                    flag_options=flag_options, write=False)
    xh = ionized_box.xH_box
    brightness_temp = p21c.brightness_temperature(ionized_box=ionized_box, perturbed_field=perturbed_field)

    np.save(f'method_{p21c.global_params.FIND_BUBBLE_ALGORITHM}_xH_z_{zre_mean}_random_seed_{random_seed}', xh)
    np.save(f'method_{p21c.global_params.FIND_BUBBLE_ALGORITHM}_density_field_z_{zre_mean}_random_seed_{random_seed}', density_field)
    np.save(f'method_{p21c.global_params.FIND_BUBBLE_ALGORITHM}_brightness_temp_z_{zre_mean}_random_seed_{random_seed}', brightness_temp.brightness_temp)



class input_info_field:
  def __init__(self):
    pass

  def set_zreion(self, P_k_zre, ion_hist, alpha, b_0, k_0):
    self.zreioninfo = self.zreioninfo(P_k_zre, ion_hist, alpha, b_0, k_0)

  def set_21cmFAST(self, P_k_zre, ion_hist, P_k_dm, z_mean, b_mz):
    self.cmFASTinfo = self.cmFASTinfo(P_k_zre, ion_hist, P_k_dm, z_mean, b_mz)

  class cmFASTinfo:
        def __init__(self, P_k_zre, ion_hist, P_k_dm, z_mean, b_mz):
            '''
            This class stores the information of the density and redshfit of reionization fields for the variational range study for 21cmFAST
            :param P_k_zre: [arr] 1D the power_spectrum of the redshift of reionization field
            :param ion_hist: [arr] 1D the ionization history
            :param P_k_dm: [arr] 1D the power_spectrum of the density  field
            :param z_mean: [float] the mean redshift of reionioization
            :param b_mz: [arr] 1D the linear bias factor
            '''
            self.P_k_zre = P_k_zre
            self.ion_hist = ion_hist
            self.P_k_dm = P_k_dm
            self.z_mean = z_mean
            self.b_mz = b_mz
        def add_brightness_temp(self, brightnesstemp, z_for_bt):
            self.z_for_bt = z_for_bt
            self.brightnesstemp = brightnesstemp
  class zreioninfo:
        def __init__(self, P_k_zre, ion_hist, alpha, b_0, k_0):
            '''
            This class stores the information of the density and redshfit of reionization fields for the variational range study for
            :param P_k_zre: [arr] 1D the power_spectrum of the redshift of reionization field
            :param ion_hist: [arr] 1D the ionization history
            :param alpha: [float] the value the alpha parameter
            :param b_0: [float] the value of the b_0 parameter
            :param k_0: [float] the value of the k_0 parameter
            '''
            self.P_k_zre = P_k_zre
            self.ion_hist = ion_hist
            self.alpha= alpha
            self.b_0 = b_0
            self.k_0 = k_0
        def add_brightness_temp(self, brightnesstemp, z_for_bt):
            self.z_for_bt = z_for_bt
            self.brightnesstemp = brightnesstemp

def analyze_float_value(obj,model,observable, Xrange, Yrange, field_names = ['Tvir','Heff']):
    '''
    This function look at the 2D variational range of a given parameter given an 2D array filled with objects
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param model: [string] the name of the analyzed model (21cmFAST or zreion)
    :param observable: [string] the name of the analyzed field (like z_mean or alpha parameter)
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :return: a 2D contour plot of the given field
    '''
    X, Y = np.meshgrid(Xrange, Yrange)
    obj_field = np.ones((len(Xrange),len(Yrange)))
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            if observable == 'ion_hist': obj_field[i][j] = pp.compute_tau(getattr(getattr(obj[i][j], f'{model}info'), observable), redshifts=np.linspace(5,15,len(getattr(getattr(obj[i][j], f'{model}info'), observable))))
            else : obj_field[i][j] = getattr(getattr(obj[i][j], f'{model}info'), observable)
    fig, ax = plt.subplots()
    plt.contourf(X,Y, obj_field)
    plt.colorbar()
    ax.set_xlabel(field_names[0])
    ax.set_ylabel(field_names[1])
    plt.title(f'{observable} variational range for {model}')
    plt.show()

def plot_variational_bias(obj,model,observable, Xrange, Yrange, xaxis=np.logspace(np.log10(0.08570025), np.log10(7.64144032), 20), add_zreion = False,  field_names = ['Tvir','Heff'], log_scale = False):
    '''
    This function plots the power spectrum over the 2D variational range of input parameters
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param field_name: [string] the name of the field to analyze
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param xaxis: [arr] the x axis array (default is k range for 143³ box)
    :param add_zreion: [bool] add the zreion bias if True
    :param log_scale: [bool] return log scale if True
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :return: a 2D contour plot of the given field
    '''

    fig, ax = plt.subplots(10,10, sharex = True, sharey = True)
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            ax[i,j].plot(xaxis, getattr(getattr(obj[i][j], f'{model}info'), observable))
            if add_zreion:
                    linbias = sa.lin_bias(xaxis, getattr(getattr(obj[i][j], f'zreioninfo'), 'alpha'), getattr(getattr(obj[i][j], f'zreioninfo'), 'b_0'), getattr(getattr(obj[i][j], f'zreioninfo'), 'k_0'))
                    ax[i,j].plot(xaxis,linbias)

    #ax.set_xlabel(field_names[0])
    #ax.set_ylabel(field_names[1])
    if log_scale: plt.loglog()
    fig.text(0.5, 0.04, field_names[0], ha='center')
    fig.text(0.04, 0.5, field_names[1], va='center', rotation='vertical')
    plt.show()


def plot_variational_PS(obj,model,observable, Xrange, Yrange, xaxis=np.logspace(np.log10(0.08570025), np.log10(7.64144032), 20), add_zreion = False, delta2 = False, field_names = ['Tvir','Heff'], log_scale = False):
    '''
    This function plots the power spectrum over the 2D variational range of input parameters. If you want to plot the 2 mdoels togheter, use cmFASt as model with the option add zreion
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param field_name: [string] the name of the field to analyze
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param xaxis: [arr] the x axis array (default is k range for 143³ box)
    :param add_zreion: [bool] add the zreion bias if True
    :param log_scale: [bool] return log scale if True
    :param delta2: [bool] compute the delta square instead of regular powe spectrum if True (P(k)*k³ / (2*pi²))
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :return: a 2D contour plot of the given field
    '''

    fig, ax = plt.subplots(10,10, sharex = True, sharey = True)
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            cmFastPP = getattr(getattr(obj[i][j], f'{model}info'), observable)[2:]
            cmFastPP /= 143**3 #temporary normalization constant (V³)
            if add_zreion:
                    zreion_PP = getattr(getattr(obj[i][j], f'zreioninfo'), observable)[1:]
                    zreion_PP /= 143**3
            if delta2:
                cmFastPP = cmFastPP * (xaxis**3) / (np.pi**2 * 2)
                if add_zreion: zreion_PP = zreion_PP * (xaxis**3) / (np.pi**2 * 2)
            ax[i, j].plot(xaxis, cmFastPP, label = '21cmFAST')
            if add_zreion: ax[i,j].plot(xaxis, zreion_PP, label = 'z-reion')

    #ax.set_xlabel(field_names[0])
    #ax.set_ylabel(field_names[1])
    if log_scale: plt.loglog()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:2], labels[:2])

    fig.text(0.5, 0.04, field_names[0], ha='center')
    fig.text(0.04, 0.5, field_names[1], va='center', rotation='vertical')
    plt.show()


def plot_variational_ion_hist(obj,model,observable, Xrange, Yrange, xaxis='redshifts', add_zreion = False, plot_diff = False,  field_names = ['Tvir','Heff'], log_scale = False):
    '''
    This function plots the power spectrum over the 2D variational range of input parameters.
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param field_name: [string] the name of the field to analyze
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param xaxis: [arr] the x axis array (default is k range for 143³ box)
    :param add_zreion: [bool] add the zreion bias if True
    :param log_scale: [bool] return log scale if True
    :param plot_diff: [bool] plot the differences in the ioniozation history from the 2 models instead of the 2 individuals ioniozation histories.
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :return: a 2D contour plot of the given field
    '''

    fig, ax = plt.subplots(10,10, sharex = True, sharey = True)
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            #if xaxis == 'redshifts': xaxis = np.linspace(5, 15,len(getattr(getattr(obj[i][j], f'{model}info'), observable)))
            if xaxis == 'redshifts': xaxis = np.linspace(5, 15, 60, endpoint=True)
            if plot_diff:
                ax[i,j].plot(xaxis, np.array(getattr(getattr(obj[i][j], f'{model}info'), observable)) - np.array(getattr(getattr(obj[i][j], f'zreioninfo'), observable)))
            else:
                ax[i,j].plot(xaxis, getattr(getattr(obj[i][j], f'{model}info'), observable), label = f'{model}')
                if add_zreion:
                        ax[i,j].plot(xaxis, getattr(getattr(obj[i][j], f'zreioninfo'), observable), label = 'z-reion')


    #ax.set_xlabel(field_names[0])
    #ax.set_ylabel(field_names[1])
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:2], labels[:2])

    if log_scale: plt.loglog()
    fig.text(0.5, 0.04, field_names[0], ha='center')
    fig.text(0.04, 0.5, field_names[1], va='center', rotation='vertical')
    plt.show()


""" The following functions comes from a code designed by P.hD candidate Lisa McBride and Prof. Paul Laplante, and credits """