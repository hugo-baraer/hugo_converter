"""
  EoR_research/tested_things.py

  Author : Hugo Baraer (including some functions by Lisa McBride and Paul Laplante at the end)
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
from astropy.cosmology import Planck15
import zreion as zr
import imageio
import powerbox as pbox
import z_re_field as zre


def compare_reion_hist(zremean_Hugo, alpha_hugo, b_0_Hugo, k_0_Hugo, ion_rates, james_zre_means, james_alphas,
                       james_k_0, plot=True, saveforgif=False, filenames=[], imnb=0, title=''):
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
    zre_zreion_me = zr.apply_zreion(density_field, zremean_Hugo, alpha_hugo, k_0_Hugo, 143, b0=b_0_Hugo)
    perturbed_field = p21c.perturb_field(redshift=james_zre_means, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_james = zr.apply_zreion(density_field, james_zre_means, james_alphas, james_k_0, 143)
    reion_hist_zreion_me = pp.reionization_history(np.linspace(5, 15, 100), zre_zreion_me, plot=False)
    reion_hist_zreion_James = pp.reionization_history(np.linspace(5, 15, 100), zre_zreion_james, plot=False)

    # pp.plot_21zreion_ionhist([reion_hist_zreion_me, ion_rates, reion_hist_zreion_James], saveforgif=saveforgif)

    return pp.plot_21zreion_ionhist([reion_hist_zreion_me, ion_rates, reion_hist_zreion_James], saveforgif=True,
                                    filenames=filenames, imnb=imnb, title=title)


def plot_variational_range_James(dict1, james_alpha, james_k_0, varying_name='Heff', varying_title='Heff'):
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


def compute_zre_zreion(alpha, b_0, k_0, zre_mean=8.011598437878444):
    '''
    This function computes the redshfit of reionization field from zreion for my  values given a set of parameters
    :param alpha: my value
    :param b_0: ""
    :param k_0:""
    :return: the redshfit of reionization field from zreion [3D array]
    '''

    # z_re = np.load('zre.npy')
    box_dim = 143  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    box_len = 143  # int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
    initial_conditions = p21c.initial_conditions(user_params=user_params, cosmo_params=cosmo_params)
    perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_me = zr.apply_zreion(density_field, 8.011598437878444, alpha, k_0, 143, b0=b_0)
    return zre_zreion_me


def compute_zre_james_me(alpha, b_0, k_0, james_alpha, james_k_0, james_mean):
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
    # z_re = np.load('zre.npy')
    box_dim = 143  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    box_len = 143  # int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
    initial_conditions = p21c.initial_conditions(user_params=user_params, cosmo_params=cosmo_params)
    perturbed_field = p21c.perturb_field(redshift=8.011598437878444, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_me = zr.apply_zreion(density_field, 8.011598437878444, alpha, k_0, 143, b0=b_0)
    perturbed_field = p21c.perturb_field(redshift=james_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    zre_zreion_james = zr.apply_zreion(density_field, james_mean, james_alpha, james_k_0, 143)
    return zre_zreion_me, zre_zreion_james


def compute_field_Adrian(zre_mean, initial_conditions, astro_params, flag_options, random_seed=12345,
                         density_field=True, xH_field=True, BT_field=True):
    '''
    This functions computes the desired field for Adrian (density field, xH field and brighness temperature field and save them as npz files)
    :param zre_mean: [float] the mean redshfit of reionization computed at
    :param initial_conditions: [obj] the 21cmFAST initial conditions objects
    :param astro_params: [obj] the astro input parameter of 21cmFASt
    :param flag_options: [obj] the flag options input parameter of 21cmFASt
    :param random_seed: [int] the random seed at wich to compute the field (default is 12345)
    :param density_field: [bool] will save the density_field if True (default True)
    :param xH_field: [bool] will save the xH (neutral hydrogen) field if True (default True)
    :param BT_field: [bool] will save the brightness temperature field if True (default True)
    :return:
    '''

    # print(density_field == p21c.run_coeval(redshift = zre_mean, user_params = user_params,cosmo_params = cosmo_params, astro_params = astro_params).density)
    if density_field:
        perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
        density_field = perturbed_field.density
        np.save(
            f'method_{p21c.global_params.FIND_BUBBLE_ALGORITHM}_density_field_z_{zre_mean}_random_seed_{random_seed}_dim200_len300Mpc',
            density_field)

    if xH_field:
        ionized_box = p21c.ionize_box(redshift=zre_mean, init_boxes=initial_conditions, astro_params=astro_params,
                                      flag_options=flag_options, write=False)
        xh = ionized_box.xH_box
        np.save(
            f'method_{p21c.global_params.FIND_BUBBLE_ALGORITHM}_xH_z_{zre_mean}_random_seed_{random_seed}_dim200_len300Mpc',
            xh)
    if BT_field:
        brightness_temp = p21c.brightness_temperature(ionized_box=ionized_box, perturbed_field=perturbed_field)
        np.save(
            f'method_{p21c.global_params.FIND_BUBBLE_ALGORITHM}_brightness_temp_z_{zre_mean}_random_seed_{random_seed}_dim200_len300Mpc',
            brightness_temp.brightness_temp)


class input_info_field:
    def __init__(self):
        pass

    def set_zreion(self, P_k_zre, ion_hist, alpha, b_0, k_0):
        self.zreioninfo = self.zreioninfo(P_k_zre, ion_hist, alpha, b_0, k_0)

    def set_21cmFAST(self, P_k_zre, ion_hist, P_k_dm, z_mean, b_mz):
        self.cmFASTinfo = self.cmFASTinfo(P_k_zre, ion_hist, P_k_dm, z_mean, b_mz)

    def set_James(self, alpha, z_mean, k_0):
        self.Jamesinfo = self.Jamesinfo(alpha, z_mean, k_0)

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

        def add_brightness_temp_mean(self, brightnesstemp_mean, std):
            self.bt_mean = brightnesstemp_mean
            self.bt_std = std

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
            self.alpha = alpha
            self.b_0 = b_0
            self.k_0 = k_0

        def add_brightness_temp(self, brightnesstemp, z_for_bt):
            self.z_for_bt = z_for_bt
            self.brightnesstemp = brightnesstemp

        def add_brightness_temp_mean(self, brightnesstemp_mean, std):
            self.bt_mean = brightnesstemp_mean
            self.bt_std = std

    class Jamesinfo:
        def __init__(self, alpha, z_mean, k_0):
            '''
            This class stores the information of the density and redshfit of reionization fields for the variational range study for
            :param alpha: [float] the value the alpha parameter
            :param z_mean: [float] the value of the mean redshfit of reionization
            :param k_0: [float] the value of the k_0 parameter
            '''
            self.alpha = alpha
            self.z_mean = z_mean
            self.k_0 = k_0

        def add_brightness_temp(self, brightnesstemp, z_for_bt):
            self.z_for_bt = z_for_bt
            self.brightnesstemp = brightnesstemp

        def add_brightness_temp_mean(self, brightnesstemp_mean, std):
            self.bt_mean = brightnesstemp_mean
            self.bt_std = std

        def add_ion_hist(self, ion_hist):
            self.ion_hist = ion_hist

        def add_P_k_zre(self, pk_zre):
            self.P_k_zre = pk_zre


def add_zreion_bt(object, redshifts, Xrange, Yrange, initial_conditions):
    '''
    computes the brightness temperature for z-reion and add it to the object
    :param object: [obj] the object containing information for
    :param redshifts: [arr] 1D array of redshfit at which to compute the redshift of reionization
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :return: [obj] 3D object containing z-reion brightness temperature for several redhfits
    '''
    for i in tqdm(range(len(Xrange)), 'computing the brigthness temperature for z-reion'):
        for j in range(len(Yrange)):
            b_temp_ps = []
            for redshift in redshifts:
                perturbed_field = p21c.perturb_field(redshift=redshift, init_boxes=initial_conditions, write=False)
                zre_zreion = zr.apply_zreion(perturbed_field.density,
                                             getattr(getattr(object[i][j], f'Jamesinfo'), 'z_mean'),
                                             getattr(getattr(object[i][j], f'zreioninfo'), 'alpha'),
                                             getattr(getattr(object[i][j], f'zreioninfo'), 'k_0'),
                                             143,
                                             b0=getattr(getattr(object[i][j], f'zreioninfo'), 'b_0'))
                ion, brightness_temp = get_21cm_fields(redshift, zre_zreion, perturbed_field.density)
                brightness_temp_ps = pbox.get_power(brightness_temp, 143, bins=20, log_bins=True)[0]
                b_temp_ps.append(brightness_temp_ps)
            object[i][j].zreioninfo.add_brightness_temp(b_temp_ps, redshifts)
    return object


def add_James_bt(object, redshifts, Xrange, Yrange, initial_conditions):
    '''
    computes the brightness temperature for the James TAU and add it to the object
    Please not that James algortithm (as compared to mine and 21cmFAST) uses Mpc / h instead of regular Mpc
    :param object: [obj] the object containing information for
    :param redshifts: [arr] 1D array of redshfit at which to compute the redshift of reionization
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :return: [obj] 3D object containing z-reion brightness temperature for several redhfits
    '''
    for i, Heff in tqdm(enumerate(Xrange), 'computing the brigthness temperature for z-reion'):
        for j, Tvir in enumerate(Yrange):
            # astro_params = p21c.AstroParams({"NU_X_THRESH": 500, "HII_EFF_FACTOR": Heff, "ION_Tvir_MIN": Tvir})
            b_temp_ps = []
            b_temp_mean = []
            b_temp_std = []
            for redshift in redshifts:
                perturbed_field = p21c.perturb_field(redshift=redshift, init_boxes=initial_conditions, write=False)
                zre_zreion = zr.apply_zreion(perturbed_field.density,
                                             getattr(getattr(object[i][j], f'Jamesinfo'), 'z_mean'),
                                             getattr(getattr(object[i][j], f'Jamesinfo'), 'alpha'),
                                             getattr(getattr(object[i][j], f'Jamesinfo'), 'k_0'),
                                             100,
                                             )
                ion, brightness_temp = get_21cm_fields(redshift, zre_zreion, perturbed_field.density)
                brightness_temp_ps = pbox.get_power(brightness_temp, 100, bins=20, log_bins=True)[0][1:]
                brightness_temp_ps /= ((100 / 143) ** 3)
                b_temp_ps.append(brightness_temp_ps)
                b_temp_mean.append(np.mean(brightness_temp))
                b_temp_std.append(np.std(brightness_temp))
            object[i][j].Jamesinfo.add_brightness_temp(b_temp_ps, redshifts)
            object[i][j].Jamesinfo.add_brightness_temp_mean(b_temp_mean, b_temp_std)
    return object


def add_James_pk_zre(object, Xrange, Yrange, initial_conditions):
    '''
    computes the brightness temperature for the James TAU and add it to the object
    Please not that James algortithm (as compared to mine and 21cmFAST) uses Mpc / h instead of regular Mpc
    :param object: [obj] the object containing information for
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :return: [obj] 2D object containing James' redshfit of reionization power_spectrum
    '''
    for i, Heff in tqdm(enumerate(Xrange), 'computing the brigthness temperature for z-reion'):
        for j, Tvir in enumerate(Yrange):
            perturbed_field = p21c.perturb_field(redshift=getattr(getattr(object[i][j], f'Jamesinfo'), 'z_mean'),
                                                 init_boxes=initial_conditions, write=False)
            zre_zreion = zr.apply_zreion(perturbed_field.density,
                                         getattr(getattr(object[i][j], f'Jamesinfo'), 'z_mean'),
                                         getattr(getattr(object[i][j], f'Jamesinfo'), 'alpha'),
                                         getattr(getattr(object[i][j], f'Jamesinfo'), 'k_0'),
                                         100,
                                         )
            # Just a little comment on unit here: 100 is used because James algorithm uses Mpc instead of Mpc/h
            pk_zre_ps = pbox.get_power(zre_zreion, 100, bins=20, log_bins=True)[0][1:]
            pk_zre_ps /= ((100 / 143) ** 3)
            # pk_zre_ps2 = pbox.get_power(zre_zreion, 143, bins=20, log_bins=True)[0]
            # print('this is pkzre1\n',pk_zre_ps)
            # print('This is pkzre2\n')
            # print(pk_zre_ps2, 'This is the diff\n')
            # print(pk_zre_ps2 - (pk_zre_ps/((100/143)**3)))
            object[i][j].Jamesinfo.add_P_k_zre(pk_zre_ps)
    return object


def add_James_ion_hist(object, Xrange, Yrange, initial_conditions, redshifts=np.linspace(5, 18, 60)):
    '''
    computes the brightness temperature for the James TAU and add it to the object
    :param object: [obj] the object containing information for
    :param redshifts: [arr] 1D array of redshfit at which to compute the redshift of reionization
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :return: [obj] 3D object containing z-reion brightness temperature for several redhfits
    '''
    for i in tqdm(range(len(Xrange)), 'computing the brigthness temperature for z-reion'):
        for j in range(len(Yrange)):
            perturbed_field = p21c.perturb_field(redshift=getattr(getattr(object[i][j], f'Jamesinfo'), 'z_mean'),
                                                 init_boxes=initial_conditions, write=False)
            zre_zreion = zr.apply_zreion(perturbed_field.density,
                                         getattr(getattr(object[i][j], f'Jamesinfo'), 'z_mean'),
                                         getattr(getattr(object[i][j], f'Jamesinfo'), 'alpha'),
                                         getattr(getattr(object[i][j], f'Jamesinfo'), 'k_0'),
                                         100,
                                         )
            zreion_hist = pp.reionization_history(redshifts, zre_zreion)
            object[i][j].Jamesinfo.add_ion_hist(zreion_hist)
    return object


def analyze_float_value(obj, model, observable, Xrange, Yrange,
                        field_names=[r'Virial temperature [$log_{10}(K)$]', 'Ionizing efficiency'], redshit_bt=15,
                        savefig=False, filenames=[]):
    '''
    This function look at the 2D variational range of a given parameter given an 2D array filled with objects
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param model: [string] the name of the analyzed model (21cmFAST or zreion)
    :param observable: [string] the name of the analyzed field (like z_mean or alpha parameter)
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :param redshit_bt: [int] the index nb of the redshfit observes if chosen 1 point statistics for the brightness temperature
    :param savefig: [bool] will save the fig and append the filename to the returned list if True. is usefull when making movies.
    :param filenames: [list] A list of filenames to append the filename of the saved figure to.
    :return: a 2D contour plot of the given field
    '''
    X, Y = np.meshgrid(Xrange, Yrange)
    obj_field = np.ones((len(Xrange), len(Yrange)))
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            if observable == 'ion_hist':
                obj_field[i][j] = pp.compute_tau(getattr(getattr(obj[i][j], f'{model}info'), observable),
                                                 redshifts=np.linspace(5, 15,
                                                                       len(getattr(getattr(obj[i][j], f'{model}info'),
                                                                                   observable))))
            elif observable == 'brightness temperature mean':
                obj_field[i][j] = getattr(getattr(obj[i][j], f'{model}info'), 'bt_mean')[redshit_bt]
            elif observable == 'brightness temperature std':
                obj_field[i][j] = getattr(getattr(obj[i][j], f'{model}info'), 'bt_std')[redshit_bt]
            else:
                obj_field[i][j] = getattr(getattr(obj[i][j], f'{model}info'), observable)

    fig, ax = plt.subplots()
    plt.contourf(X, Y, obj_field)
    plt.colorbar()
    ax.set_xlabel(field_names[0])
    ax.set_ylabel(field_names[1])
    if observable == 'ion_hist': observable = 'TAU'
    if observable == 'brightness temperature mean' or observable == 'brightness temperature std': plt.title(
        f'{observable} variational range for {model} at a redshift of {redshit_bt}')
    plt.title(f'{observable} variational range for {model}')
    if savefig:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.savefig('./bt_map/bt1James_z{}.png'.format(redshit_bt))
        filenames.append('./bt_map/bt1James_z{}.png'.format(redshit_bt))
        plt.close()
        return filenames
    else:
        plt.show()
        return obj_field


def analyze_frac_diff(obj, model1, model2, observable, Xrange, Yrange,
                      field_names=[r'Virial temperature [$log_{10}(K)$]', 'Ionizing efficiency'], redshift_bt=None):
    '''
    This function lookat the fractional difference between the observable of 2  models. It works for the TAU parameter and the brightness temeprature means or stds. (Model1- Model2 / Model1) You can make it work for other params if you take out the [redshfit_bt] from the else function
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param model1: [string] the name of the analyzed model (21cmFAST, James or zreion)
    :param model2: [string] the name of the analyzed model (21cmFAST, James or zreion)
    :param observable: [string] the name of the analyzed field (like z_mean or alpha parameter)
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :param redshift_bt: [int] the slice of brightness temperature corresponding to the desired redshift
    :return: a 2D contour plot of the fractional difference for the given field
    '''
    X, Y = np.meshgrid(Xrange, Yrange)
    obj_field = np.ones((len(Xrange), len(Yrange)))
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            if observable == 'ion_hist':
                cmFAST_TAU = pp.compute_tau(getattr(getattr(obj[i][j], f'{model1}info'), observable),
                                            redshifts=np.linspace(5, 18,
                                                                  len(getattr(getattr(obj[i][j], f'{model1}info'),
                                                                              observable))))
                zreion_TAU = pp.compute_tau(getattr(getattr(obj[i][j], f'{model2}info'), observable),
                                            redshifts=np.linspace(5, 18,
                                                                  len(getattr(getattr(obj[i][j], f'{model1}info'),
                                                                              observable))))
                diff_TAU = cmFAST_TAU - zreion_TAU

                obj_field[i][j] = diff_TAU / cmFAST_TAU
                if obj_field[i][j] > 0.08: obj_field[i][j] = 0.08
            else:
                cmFAST_TAU = getattr(getattr(obj[i][j], f'{model1}info'), observable)[redshift_bt]
                zreion_TAU = getattr(getattr(obj[i][j], f'{model2}info'), observable)[redshift_bt]
                diff_TAU = cmFAST_TAU - zreion_TAU
                print(diff_TAU)
                obj_field[i][j] = diff_TAU / cmFAST_TAU
                # if obj_field[i][j] < -0.4 : obj_field[i][j] = -0.4

    fig, ax = plt.subplots()
    if observable == 'ion_hist':
        levels = np.linspace(obj_field.min(), 0.08, 3000)
        cntr = plt.contourf(X, Y, obj_field, levels=levels, vmin=-0.06, vmax=0.06, cmap='RdBu')
    else:
        levels = np.linspace(obj_field.min(), obj_field.max(), 3000)
        absolute_big = max(obj_field.max(), abs(obj_field.min()))
        if absolute_big > 1: absolute_big = 1
        cntr = plt.contourf(X, Y, obj_field, levels=levels, vmin=-absolute_big, vmax=absolute_big, cmap='RdBu')
    # plt.clim(-0.06, 0.06)
    plt.colorbar(cntr, ax=ax)
    ax.set_xlabel(field_names[0])
    ax.set_ylabel(field_names[1])
    if model1 == 'zreion': model1 = 'Hugo converter'
    if model2 == 'zreion': model2 = 'Hugo converter'
    if observable == 'ion_hist':
        plt.title(f'TAU variational range for ({model1} - {model2}) / {model1}')
    else:
        plt.title(
            f'{observable} variational range for ({model1} - {model2}) / {model1} at a redshfit z = {obj[0][0].cmFASTinfo.z_for_bt[redshift_bt]}')
    plt.show()


def plot_variational_bias(obj, model, observable, Xrange, Yrange,
                          xaxis=np.logspace(np.log10(0.08570025), np.log10(7.64144032), 20), add_zreion=False,
                          field_names=['Tvir', 'Heff'], log_scale=False):
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

    fig, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            ax[i, j].plot(xaxis, getattr(getattr(obj[i][j], f'{model}info'), observable), label='21cmFAST')
            if add_zreion:
                linbias = sa.lin_bias(xaxis, getattr(getattr(obj[i][j], f'zreioninfo'), 'alpha'),
                                      getattr(getattr(obj[i][j], f'zreioninfo'), 'b_0'),
                                      getattr(getattr(obj[i][j], f'zreioninfo'), 'k_0'))
                ax[i, j].plot(xaxis, linbias, label='z-reion')

    # ax.set_xlabel(field_names[0])
    # ax.set_ylabel(field_names[1])
    if log_scale: plt.loglog()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:2], labels[:2])
    fig.text(0.45, 0.9, r'Linear bias $b_{mz}$ as a function of Heff and Tvir ', size='large')
    fig.text(0.5, 0.02, field_names[0], ha='center')

    fig.text(0.5, 0.04, '                 3.8'
                        '                              3.9'
                        '                            4.0'
                        '                            4.1'
                        '                            4.2'
                        '                            4.3'
                        '                            4.4'
                        '                            4.5'
                        '                            4.6'
                        '                            4.7', ha='center')
    fig.text(0.04, 0.5, field_names[1], va='center', rotation='vertical')
    fig.text(0.06, 0.5, '25'
                        '             28'
                        '             31'
                        '             34'
                        '             37'
                        '             40'
                        '             43'
                        '             46'
                        '             49'
                        '             52', va='center', rotation='vertical')
    plt.show()


def plot_variational_bright_temp(obj, model, observable, redshift, slice, Xrange, Yrange,
                                 xaxis=np.logspace(np.log10(0.08570025), np.log10(7.64144032), 20), add_zreion=False,
                                 field_names=['Tvir', 'Heff'], log_scale=True, savefig=False, filenames=[],
                                 add_James=False,
                                 xaxis_James=np.logspace(np.log10(0.08885766), np.log10(6.45765192), 20)):
    '''
    This function plots the power spectrum over the 2D variational range of input parameters
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param observable: [string] the observable to analyze (power spectrum in this case)
    :param redshift: [float] the redshift at which the brightness temperature is analyzed
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param xaxis: [arr] the x axis array (default is k range for 143³ box)
    :param add_zreion: [bool] add the zreion bias if True
    :param log_scale: [bool] return log scale if True
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :param savefig: [bool] will save the fig and append the filename to the returned list if True. is usefull when making movies.
    :param filenames: [list] A list of filenames to append the filename of the saved figure to.
    :return: a 2D contour plot of the given field
    '''

    fig, ax = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(19, 9.5))
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            cmFastPP = getattr(getattr(obj[i][j], f'{model}info'), observable)[slice]
            ax[i, j].plot(xaxis, cmFastPP, label='21cmFAST')
            ax[i, j].set_xlabel(r'k [$\frac{1}{Mpc}$]')

            if j == 0:
                ax[i, j].set_ylabel(r'$P_{bt}$ [K²Mpc³]', size='small')
            if add_zreion:
                linbias = getattr(getattr(obj[i][j], f'zreioninfo'), observable)[slice]
                ax[i, j].plot(xaxis, linbias, label='Hugo converter')
            if add_James:
                James_PP = getattr(getattr(obj[i][j], f'Jamesinfo'), observable)[slice]
                # James_PP_correctedh = James_PP / (0.7 ** 3)
                ax[i, j].plot(xaxis, James_PP, label='James algorithm')
                # James_PP2 = getattr(getattr(obj[i+1][j+1], f'Jamesinfo'), observable)[slice]
                # print(James_PP2-James_PP)
            # print(cmFastPP/linbias)
    # ax.set_xlabel(field_names[0])
    # ax.set_ylabel(field_names[1])
    # plt.ylim([-324049718.6696194, 6844765738.942251])
    if log_scale: plt.loglog()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3])

    plt.ylim([1198742.8859540655, 9818689819.912655])
    # plt.title(r'Linear bias $b_mz$ as a function of Heff and Tvir ')
    fig.text(0.5, 0.01, r'Virial temperature [$log_{10}(K)$]', ha='center')
    fig.text(0.4, 0.9, f'Power spectrum of the Brightness temperature at redshift z = {redshift} ', size='large')
    Yrange = [round(item, 2) for item in Yrange]
    Xrange = [round(item, 2) for item in Xrange]
    fig.text(0.5, 0.032
             , f'                 {Xrange[0]}'
               f'                              {Xrange[1]}'
               f'                            {Xrange[2]}'
               f'                            {Xrange[3]}'
               f'                            {Xrange[4]}'
               f'                            {Xrange[5]}'
               f'                            {Xrange[6]}'
               f'                            {Xrange[7]}'
               f'                            {Xrange[8]}'
               f'                            {Xrange[9]}', ha='center')
    fig.text(0.04, 0.5, 'Ionizing efficiency', va='center', rotation='vertical')
    fig.text(0.06, 0.5, f'{Yrange[0]}'
                        f'             {Yrange[1]}'
                        f'             {Yrange[2]}'
                        f'             {Yrange[3]}'
                        f'             {Yrange[4]}'
                        f'             {Yrange[5]}'
                        f'             {Yrange[6]}'
                        f'             {Yrange[7]}'
                        f'             {Yrange[8]}'
                        f'             {Yrange[9]}', va='center', rotation='vertical')
    if savefig:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.savefig('./bt_map/bt1James_z{}.png'.format(redshift))
        filenames.append('./bt_map/bt1James_z{}.png'.format(redshift))
        plt.close()
        return filenames
    else:
        plt.show()


def plot_variational_PS(obj, model, observable, Xrange, Yrange, redshift=None, slice=None,
                        xaxis=np.logspace(np.log10(0.08570025), np.log10(7.64144032), 19), add_zreion=False,
                        add_James=False, delta2=False,
                        field_names=[r'Virial temperature [$log_{10}(K)$]', 'Ionizing efficiency'], log_scale=True,
                        savefig=False,
                        filenames=[]):
    '''
    This function plots the power spectrum over the 2D variational range of input parameters. If you want to plot the 2 mdoels togheter, use cmFASt as model with the option add zreion
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param observable: [string] the name of the field to analyze
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param xaxis: [arr] the x axis array (default is k range for 143³ box)
    :param add_zreion: [bool] add the zreion bias if True
    :param log_scale: [bool] return log scale if True
    :param delta2: [bool] compute the delta square instead of regular powe spectrum if True (P(k)*k³ / (2*pi²))
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :return: a 2D contour plot of the given field
    '''

    fig, ax = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(19, 9.5))
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            cmFastPP = getattr(getattr(obj[i][j], f'{model}info'), observable)
            # cmFastPP /= 143**3 #temporary normalization constant (V³)
            if add_zreion:
                zreion_PP = getattr(getattr(obj[i][j], f'zreioninfo'), observable)
            if add_James:
                James_PP = getattr(getattr(obj[i][j], f'Jamesinfo'), observable)
            if observable == 'brightnesstemp':
                cmFastPP = cmFastPP[slice]
                if add_zreion: zreion_PP = zreion_PP[slice]
                if add_James: James_PP = James_PP[slice]
            if delta2:
                cmFastPP = cmFastPP * (xaxis ** 3) / (np.pi ** 2 * 2)
                if add_zreion: zreion_PP = zreion_PP * (xaxis ** 3) / (np.pi ** 2 * 2)
                if add_James: James_PP = James_PP * (xaxis ** 3) / (np.pi ** 2 * 2)
            ax[i, j].plot(xaxis, cmFastPP, label='21cmFAST')
            ax[i, j].set_xlabel(r'k [$\frac{1}{Mpc}$]')
            if add_zreion: ax[i, j].plot(xaxis, zreion_PP, label='Hugo converter')
            if add_James: ax[i, j].plot(xaxis, James_PP, label='James algorithm')
    # ax.set_xlabel(field_names[0])
    # ax.set_ylabel(field_names[1])
    if log_scale: plt.loglog()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3])
    if observable == 'brightnesstemp':
        fig.text(0.4, 0.9, f'Power spectrum of the Brightness temperature at redshift z = {redshift} ', size='large')
    else:
        fig.text(0.45, 0.9, f'Power spectrum of the redshift of reionization field', size='large')
    # plt.title('The Power spectrum of the redshfit of reioniozation fields as a function of heff and Tvir')
    fig.text(0.5, 0.02, field_names[0], ha='center')
    Yrange = [int(item) for item in Yrange]
    Xrange = [round(item, 2) for item in Xrange]
    fig.text(0.5, 0.032
             , f'                 {Xrange[0]}'
               f'                              {Xrange[1]}'
               f'                            {Xrange[2]}'
               f'                            {Xrange[3]}'
               f'                            {Xrange[4]}'
               f'                            {Xrange[5]}'
               f'                            {Xrange[6]}'
               f'                            {Xrange[7]}'
               f'                            {Xrange[8]}'
               f'                            {Xrange[9]}', ha='center')
    fig.text(0.04, 0.5, 'Ionizing efficiency', va='center', rotation='vertical')
    fig.text(0.06, 0.5, f'{Yrange[9]}'
                        f'             {Yrange[8]}'
                        f'             {Yrange[7]}'
                        f'             {Yrange[6]}'
                        f'             {Yrange[5]}'
                        f'             {Yrange[4]}'
                        f'             {Yrange[3]}'
                        f'             {Yrange[2]}'
                        f'             {Yrange[1]}'
                        f'             {Yrange[0]}', va='center', rotation='vertical')

    if log_scale: plt.loglog()
    if savefig:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.savefig('./bt_map/bt1James_z{}.png'.format(redshift))
        filenames.append('./bt_map/bt1James_z{}.png'.format(redshift))
        plt.close()
        return filenames
    else:
        plt.show()


def plot_variational_ion_hist(obj, model, observable, Xrange, Yrange, xaxis='redshifts', add_zreion=False,
                              add_James=False, plot_diff=False,
                              field_names=[r'Virial temperature [$log_{10}(K)$]', 'Ionizing efficiency'],
                              log_scale=False):
    '''
    This function plots the ionization histories over the 2D variational range of input parameters.
    :param obj: [arr] 2D, the object array filled with info of 21cmFAST and zreion
    :param observable: [string] the name of the field to analyze
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :param xaxis: [arr] the x axis array (default is k range for 143³ box)
    :param add_zreion: [bool] add the zreion histories if True
    :param add_James: [bool] add the james histories if True
    :param log_scale: [bool] return log scale if True
    :param plot_diff: [bool] plot the differences in the ioniozation history from the 2 models instead of the 2 individuals ioniozation histories.
    :param field names: [list] the 2 element list of field names (default Heff and Tvir)
    :return: a 2D 10x10 grid plot of the ionization histories
    '''

    fig, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            # if xaxis == 'redshifts': xaxis = np.linspace(5, 15,len(getattr(getattr(obj[i][j], f'{model}info'), observable)))
            if xaxis == 'redshifts': xaxis = np.linspace(5, 18, 60, endpoint=True)
            if plot_diff:
                ax[i, j].plot(xaxis, np.array(getattr(getattr(obj[i][j], f'{model}info'), observable)) - np.array(
                    getattr(getattr(obj[i][j], f'zreioninfo'), observable)))
            else:
                ax[i, j].plot(xaxis, getattr(getattr(obj[i][j], f'{model}info'), observable), label=f'{model}')
                ax[i, j].set_xlabel(r'redshift')
                if j == 0:
                    ax[i, j].set_ylabel(r'$x_i$', size='small')
                if add_zreion:
                    ax[i, j].plot(xaxis, getattr(getattr(obj[i][j], f'zreioninfo'), observable), label='Hugo algorithm')
                if add_James:
                    ax[i, j].plot(xaxis, getattr(getattr(obj[i][j], f'Jamesinfo'), observable), label='James algorithm')

    # ax.set_xlabel(field_names[0])
    # ax.set_ylabel(field_names[1])
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3])
    fig.text(0.45, 0.9, f'Ionization history as function of the ioniozation efficiency and virial temperature',
             size='large')
    if log_scale: plt.loglog()
    fig.text(0.5, 0.01, r'Virial temperature [$log_{10}(K)$]', ha='center')

    Yrange = [int(item) for item in Yrange]
    Xrange = [round(item, 2) for item in Xrange]
    fig.text(0.5, 0.032
             , f'                 {Xrange[0]}'
               f'                              {Xrange[1]}'
               f'                            {Xrange[2]}'
               f'                            {Xrange[3]}'
               f'                            {Xrange[4]}'
               f'                            {Xrange[5]}'
               f'                            {Xrange[6]}'
               f'                            {Xrange[7]}'
               f'                            {Xrange[8]}'
               f'                            {Xrange[9]}', ha='center')
    fig.text(0.04, 0.5, 'Ionizing efficiency', va='center', rotation='vertical')
    fig.text(0.06, 0.5, f'{Yrange[9]}'
                        f'             {Yrange[8]}'
                        f'             {Yrange[7]}'
                        f'             {Yrange[6]}'
                        f'             {Yrange[5]}'
                        f'             {Yrange[4]}'
                        f'             {Yrange[3]}'
                        f'             {Yrange[2]}'
                        f'             {Yrange[1]}'
                        f'             {Yrange[0]}', va='center', rotation='vertical')
    plt.show()


def add_James_params(object, jamesparams, Xrange, Yrange):
    '''
    Adds james  parameters to the variational study
    :param object: [obj] the object containing information for
    :param jamesparams: [arr] 2D array of list of James parameters [alpha, k_0, z_mean]
    :param Xrange: [arr] the 1D array of the Xrange
    :param Yrange: [arr] the 1D array of the Yrange
    :return: [obj] 2d grid of objects containing James' parameters
    '''
    for i in range(len(Xrange)):
        for j in range(len(Yrange)):
            object[i][j].set_James(jamesparams[i][j][0], jamesparams[i][j][2], jamesparams[i][j][1])
    return object


""" The following functions comes from a code designed by P.hD candidate Lisa McBride and Prof. Paul Laplante, and credits for the following function are FULLY DESERVED by these two """

omegam = Planck15.Om0
omegab = Planck15.Ob0
hubble0 = Planck15.H0


# global temperature as a function of redshift
def t0(z):
    return 38.6 * 70 * np.sqrt(
        (1 + z) / 10)  # 38.6 * hubble0.value * (omegab / 0.045) * np.sqrt(0.27 / omegam * (1 + z) / 10)


def get_21cm_fields(z, zreion, delta):
    # print("computing t21 at z=", z, "...")
    ion_field = np.where(zreion > z, 1.0, 0.0)
    t21_field = t0(z) * (1 + delta) * (1 - ion_field)

    return ion_field, t21_field


def create_reion_history(redshifts, zreion, delta, rez=143):
    neutral_frac = np.zeros_like(redshifts)
    for i, z in enumerate(redshifts):
        ion_field, t21_field = get_21cm_fields(z, zreion, delta)
        ion_frac = ion_field.sum() / rez ** 3
        neutral_frac[i] = 1 - ion_frac

    return neutral_frac
