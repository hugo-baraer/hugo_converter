"""
#  EoR_research/project_driver.py

  Author : Hugo Baraer
  Supervision by : Prof. Adrian Liu
  Affiliation : Cosmic dawn group at McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules computing the require fields and parameters.
"""

# import classic python librairies
import py21cmfast as p21c
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import emcee
import imageio
import corner
import powerbox as pbox
import zreion as zr
# import this project's modules
from . import z_re_field as zre
from . import statistical_analysis as sa
from . import plot_params as pp
from . import z_reion_comparison as zrcomp
# For testing purposes, these two modules can be installed:
# import Gaussian_testing as gauss
# import FFT


#Uncomment the following lines to import and show observables of the object
# Heff_range = np.linspace(66, 30, 10, endpoint=True)
# T_vir_range = np.linspace(4.0, 4.9, 10, endpoint=True)
# stoo = np.load('Heff30to66_Tvir40to49_withJamesall_100box.npy',
#                allow_pickle=True)  # loaded with 143, can load 100 for comparison (r even uncorrected 100

# ADD JAMES PARAMS from translator
# James_params = np.load('Heff30to66_Tvir40to49_Jamesparamsonly.npy', allow_pickle=True)
# objjj = zrcomp.add_James_params(stoo,James_params,Heff_range,T_vir_range)
# np.save('Heff30to66_Tvir40to49_withJamesparams', objjj)

# ADD JAMES IONIZATION HISTORY
# wJames_params = np.load('Heff30to66_Tvir40to49_withJamesparams.npy', allow_pickle=True)
# redshfit_4_bt = stoo[0][0].cmFASTinfo.z_for_bt
# user_params = {"HII_DIM": 143, "BOX_LEN": 143, "DIM":143}
# cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
# initial_conditions = p21c.initial_conditions(
#         user_params = user_params,
#         cosmo_params = cosmo_params
#         )
# objjj = zrcomp.add_James_ion_hist(wJames_params,Heff_range,T_vir_range,initial_conditions)
# np.save('Heff30to66_Tvir40to49_withJamesion_hist_nobt', objjj)
# raise ValueError

# ADD JAMES POWER SPECTRUMS ( ZRE FIELD OR BT FIELD)
# redshfit_4_bt = stoo[0][0].cmFASTinfo.z_for_bt
# user_params = {"HII_DIM": 143, "BOX_LEN": 143, "DIM":143}
# cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
# initial_conditions = p21c.initial_conditions(
#         user_params = user_params,
#         cosmo_params = cosmo_params
#         )
# #objjj = zrcomp.add_James_pk_zre(stoo,Heff_range,T_vir_range,initial_conditions)
# objjj = zrcomp.add_James_bt(stoo,redshfit_4_bt,Heff_range,T_vir_range,initial_conditions)
# #np.save('Heff30to66_Tvir40to49_withJamesion_hist_withpkzre2_100box_not_corrected', objjj)
# np.save('Heff30to66_Tvir40to49_withJamesall_100box', objjj)
# #
# raise ValueError


# Plot the main observables
# zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion=True, add_James=True,
#                                  xaxis=np.linspace(5, 18, 60))
# # zrcomp.plot_variational_PS(stoo,'cmFAST','P_k_zre', T_vir_range, Heff_range, add_zreion=True, add_James = True, delta2= True)
# slice = 15
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])
# zrcomp.analyze_frac_diff(stoo, 'cmFAST', 'zreion', 'ion_hist', T_vir_range, Heff_range, redshift_bt=slice)
# zrcomp.analyze_frac_diff(stoo, 'cmFAST', 'James', 'ion_hist', T_vir_range, Heff_range, redshift_bt=slice)
# zrcomp.analyze_frac_diff(stoo, 'zreion', 'James', 'ion_hist', T_vir_range, Heff_range, redshift_bt=slice)
#
# cmFAST_mean = zrcomp.analyze_float_value(stoo, 'cmFAST', 'brightness temperature mean', T_vir_range, Heff_range)
# cmFAST_std = zrcomp.analyze_float_value(stoo, 'cmFAST', 'brightness temperature std', T_vir_range, Heff_range)
# zreion_mean = zrcomp.analyze_float_value(stoo, 'zreion', 'brightness temperature mean', T_vir_range, Heff_range)
# zreion_std = zrcomp.analyze_float_value(stoo, 'zreion', 'brightness temperature std', T_vir_range, Heff_range)
#
# James_mean = zrcomp.analyze_float_value(stoo, 'James', 'brightness temperature mean', T_vir_range, Heff_range)
# James_std = zrcomp.analyze_float_value(stoo, 'James', 'brightness temperature std', T_vir_range, Heff_range)
#
# zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion=True, add_James=True,
#                                  xaxis=np.linspace(5, 18, 60))


# make a brightness temperature movie
def make_bt_movie(stoo, k_values, Heff_range, T_vir_range, gifname, add_zreion=True, add_James=True):
    '''
    This function makes a brightness temperature movie
    :param stoo: the object of 2D parameter space containing all the brightness temperature information
    :param kvalues: the values of k of the power_spectrum computations
    :param add_zreion: (default True) add z-reion info if True
    :param add_James: (default True) add James' algorithm info if True
    :param gifname: [string] the name you want to give to your gif
    :return:
    '''
    filenames = []
    for slice in tqdm(range(len(stoo[0][0].cmFASTinfo.z_for_bt)), 'making a reionization movie'):
        filenames = zrcomp.plot_variational_PS(stoo, 'cmFAST', 'brightnesstemp',
                                               T_vir_range, Heff_range, xaxis=k_values, add_zreion=add_zreion,
                                               savefig=True, filenames=filenames, add_James=add_James,
                                               redshift=stoo[0][0].cmFASTinfo.z_for_bt[slice], slice=slice,
                                               )

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{gifname}.gif', images)
    return


# zrcomp.plot_multiple_ion_hist(stoo,'zreion','ion_hist', T_vir_range, Heff_range)
# zrcomp.plot_variational_PS(stoo, 'cmFAST', 'P_k_zre', T_vir_range, Heff_range, add_zreion=True, add_James=True,delta2=True)
#make_bt_movie(stoo, np.logspace(np.log10(0.08570025), np.log10(7.64144032), 19, endpoint=True), Heff_range, T_vir_range,
              #'FINAL_bt_power_spectrum_3models', add_James=True)


# zrcomp.plot_multiple_ion_hist(stoo,'zreion','ion_hist', T_vir_range, Heff_range)

# zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, plot_diff=True, xaxis=np.linspace(5,18,60))
# zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, xaxis=np.linspace(5,18,60))
# p_k_zre, kbins_zre = pbox.get_power(stoo[0][0].cmFASTinfo.P_k_zre,143)
#
#


# zre_mean = [6.8,7.0,7.2,7.4,7.6,7.8,8.0]
# random_seed = [12345,54321,23451,34512,45123]

# random_seed = random_seed[0]

def compute_several_21cmFASt_fields(zre_mean, box_dim, box_len, astro_params={"NU_X_THRESH": 500},
                                    find_bubble_algorithm=int(2), density_field=True, xH_field=True, BT_field=True,
                                    random_seed=[54321]):
    '''
    This function generates fields and save fields at various redshift and random seeds.
    :param random_seed: [list] all the desired random seed used
    :param zre_mean: [list] the desired computation redshifts
    :param box_len: [int] the spatial length of the desired box in Mpc (default is 143 Mpc which is equivalent to 100 Mpc/h)
    :param box_dim: [int] the dimension of the box (number of points per field) (default is 143 for a spatial voxel resolution of (1 Mpc/h)³
    :param astro_params: the astro_params you want the fields to have (cosmological parameters can be changed by hand.
    :param find_bubble_algorithm: [int] The find bubble algorithm method use (default 2)
    :param density_field: [bool] will save the density_field if True (default True)
    :param xH_field: [bool] will save the xH (neutral hydrogen) field if True (default True)
    :param BT_field: [bool] will save the brightness temperature field if True (default True)
    :return: saved fields under the name :
    '''

    # run the loop through random seed and redshifts
    for count1, Heff in enumerate(tqdm(random_seed)):
        for count2, Tvir in enumerate(tqdm(zre_mean)):
            user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_dim}
            cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
            astro_params = p21c.AstroParams(
                astro_params)  # ","HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir, HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir  #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
            flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": False})
            # add astro_params
            p21c.global_params.FIND_BUBBLE_ALGORITHM = find_bubble_algorithm
            initial_conditions = p21c.initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                random_seed=Heff
            )

            zrcomp.compute_field_Adrian(Tvir, initial_conditions, astro_params, flag_options, random_seed=Heff,
                                        density_field=density_field, xH_field=xH_field, BT_field=BT_field)


# def compute21cmFAST_zre_field():
def get_params_values(box_len=143, box_dim=143, include_confidencerange=False, redshift_range=np.linspace(5, 18, 60),
                      nb_bins=20, density_field=None, zre_field=None, plot_best_fit=False, plot_corner=False,
                      return_zre_field=False, return_density=False, return_power_spectrum=False,
                      astro_params={"HII_EFF_FACTOR": 30.0}, flag_options={"USE_MASS_DEPENDENT_ZETA": False},
                      SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045, POWER_INDEX=0.9665, find_bubble_algorithm=2):
    '''
    This function computes the linear bias free parameter values of z-reion
    :param box_len: [int] the spatial length of the desired box in Mpc (default is 143 Mpc which is equivalent to 100 Mpc/h)
    :param box_dim: [int] the dimension of the box (number of points per field) (default is 143 for a spatial voxel resolution of (1 Mpc/h)³
    :param include_confidencerange: [bool] return the confidence range (upper and lower limit) of the parameters. This corresponds to 68% of the posterior distribution of each parameter
    :param redshift_range: [1D array] this is the redshift range used for the computation of the redshift of reionization. The more precise the range (the more element in the array), the more precise/accurate the values of the parameter are, but the more computational time it takes
    :param nb_bins: [int] the number of data points for the power spectrums and the bias (default 20). More can increase precision but reduce accuracy. Past work shows sweet point being the default 20
    :param density_field: [3D array] The density field used for the bias computation. None computes and uses 21cmFAST density field (default None)
    :param zre_field: [3D array] The redshift of reionization field used for the bias computation. None computes and uses 21cmFAST density field (default None)
    :param plot_best_fit: [bool] Will plot the best fitted paramters over the computed bias if True (default True)
    :param plot_corner: [bool] Will plot the posterior distribution of the best fitted parameters if True (default True)
    :param return_zre_field: [bool] will return the redshift of reionization field if True (defaut True)
    :param return_density: [bool] will return the density field if True (defaut True)
    :param return_power_spectrum: [bool] will return the power spectrums of the density field and redshift of reionization field if True (defaut True)
    :param astro_params: [dict] a dictionnary of all the wanted non-default astrophysical parameters on the form { input_param : value, ...} An extensive list of the usable astro parameters can be find here : https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :param flag_options: [dict] a dictionnary of all the wanted non-default flag options parameters on the form { flag_option : value, ...}. This include the use-mass_dependant_zeta function for the usage of astro parameters such as the turnover mass. An extensive list of the usable flag options can be find here : https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :param SIGMA_8: [float] the cosmological value (default 0.8 )
    :param hlittle: [float] the cosmological value (default 0.7)
    :param OMm: [float] the cosmological value (default 0.27)
    :param OMb: [float] the cosmological value (default 0.045)
    :param POWER_INDEX: [float] the cosmological value (default 0.9665 )
    :param find_bubble_algorithm: [int] what method to use when finding the bubbles (default = 2)
    :return: [int or list] the values for the best-fitted free parameters alpha, b_0 and k_0, plus all other optional observable (the results gives a list if confidence interval are included)
    '''

    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_dim}
    cosmo_params = p21c.CosmoParams(SIGMA_8=SIGMA_8, hlittle=hlittle, OMm=OMm, OMb=OMb)
    astro_params = p21c.AstroParams(
        astro_params)  # ","HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir, HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir  #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
    flag_options = p21c.FlagOptions(flag_options)
    # p21c.global_params.FIND_BUBBLE_ALGORITHM = find_bubble_algorithm
    # test
    initial_conditions = p21c.initial_conditions(user_params=user_params,
                                                 cosmo_params=cosmo_params)  # random_seed = Heff

    # comment after variational run
    data_dict = {'Z_re': [], "medians": [], "a16": [], "a50": [], "a84": [],
                 "b16": [], "b50": [], "b84": [], "k16": [], "k50": [], "k84": [], "p16": [], "p50": [], "p84": [],
                 "width50": [], "width90": []}

    if return_zre_field and return_density:
        data_dict, density_field, z_re_field = sa.generate_bias(redshift_range, initial_conditions, box_dim, box_len,
                                                                astro_params,
                                                                flag_options, density_field=density_field,
                                                                z_re_box=zre_field,
                                                                data_dict=data_dict, nb_bins=20,
                                                                plot_best_fit=plot_best_fit,
                                                                plot_corner=plot_corner, comp_zre_PP=False,
                                                                logbins=True,
                                                                comp_ion_hist=False, comp_bt=False,
                                                                return_zre=return_zre_field,
                                                                return_density=return_density)

    elif return_zre_field or return_density:
        data_dict, single_field = sa.generate_bias(redshift_range, initial_conditions, box_dim, box_len, astro_params,
                                                   flag_options, density_field=density_field, z_re_box=zre_field,
                                                   data_dict=data_dict, nb_bins=20, plot_best_fit=plot_best_fit,
                                                   plot_corner=plot_corner, comp_zre_PP=False, logbins=True,
                                                   comp_ion_hist=False, comp_bt=False, return_zre=return_zre_field,
                                                   return_density=return_density)
    else:
        data_dict = sa.generate_bias(redshift_range, initial_conditions, box_dim, box_len, astro_params,
                                     flag_options, density_field=density_field, z_re_box=zre_field,
                                     data_dict=data_dict, nb_bins=20, plot_best_fit=plot_best_fit,
                                     plot_corner=plot_corner, comp_zre_PP=False, logbins=True,
                                     comp_ion_hist=False, comp_bt=False, return_zre=return_zre_field,
                                     return_density=return_density)

    if include_confidencerange:
        a_range = [float(data_dict['a16'][0]), float(data_dict['a50'][0]), float(data_dict['a84'][0])]
        b_range = [float(data_dict['b16'][0]), float(data_dict['b50'][0]), float(data_dict['b84'][0])]
        k_range = [float(data_dict['k16'][0]), float(data_dict['k50'][0]), float(data_dict['k84'][0])]
    else:
        a_range = float(data_dict['a50'][0])
        b_range = float(data_dict["b50"][0])
        k_range = float(data_dict["k50"][0])

    if not return_density and not return_zre_field:
        return a_range, b_range, k_range
    elif not return_density and return_zre_field:
        return a_range, b_range, k_range, single_field
    elif not return_zre_field and return_density:
        return a_range, b_range, k_range, single_field
    elif return_density and return_zre_field:
        return a_range, b_range, k_range, density_field, z_re_field


# a, b, k, density_field = get_params_values(include_confidencerange=True,  redshift_range=np.linspace(5, 18, 15), return_density=True)

# print(a, b, k)
# print('coucou')

def params_changing_run(name_input1, range1, redshift_range=np.linspace(5, 18, 60), box_dim=143, box_len=143,
                        is_astro=True, other_astro_params={"NU_X_THRESH": 500}, find_bubble_algorithm=2,
                        flag_options={"USE_MASS_DEPENDENT_ZETA": False}, plot_best_fit=False, plot_corner=False):
    '''
    This function runs a series of 21cmFAST runs with a changing input (can be astrophysical or cosmological) and returns a dictionnary of the different parameters values (and confidence range) for the run.
    :param name_input1: [string] the name of the changing input as presented in 21cmFASt inputs list (ex: HII_EFF_FACTOR for ionization efficiency (zeta))
    :param range1: [array or list] the range of the parameter you want to check
    :param redshift_range: [1D array] this is the redshift range used for the computation of the redshift of reionization. The more precise the range (the more element in the array), the more precise/accurate the values of the parameter are, but the more computational time it takes
    :param box_len: [int] the spatial length of the desired box in Mpc (default is 143 Mpc which is equivalent to 100 Mpc/h)
    :param box_dim: [int] the dimension of the box (number of points per field) (default is 143 for a spatial voxel resolution of (1 Mpc/h)³
    :param is_astro: [bool] if the entered parameter is astrophysical (default True)
    :param other_astro_params: [dict] if you want another astrophysical parameters to stay stable, but under a different value than the default one
    :param flag_options: [dict] a dictionnary of all the wanted non-default flag options parameters on the form { flag_option : value, ...}. This include the use-mass_dependant_zeta function for the usage of astro parameters such as the turnover mass. An extensive list of the usable flag options can be find here : https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :return: [dict] a dictionnnary with all the information for the parameters and their confidence interval
    '''

    data_dict = {'Z_re': [], '{}'.format(name_input1): [], "medians": [], "a16": [], "a50": [], "a84": [],
                 "b16": [], "b50": [], "b84": [], "k16": [], "k50": [], "k84": [], "p16": [], "p50": [], "p84": [],
                 "width50": [], "width90": []}
    for count1, Heff in enumerate(tqdm(range1, 'Computing the parameters across the range')):

        user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_dim}
        flag_options = p21c.FlagOptions(flag_options)
        # p21c.global_params.FIND_BUBBLE_ALGORITHM = find_bubble_algorithm

        if is_astro:
            astro_params = other_astro_params | {f'{name_input1}': Heff}
            cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
        else:
            astro_params = other_astro_params
            if name_input1 == 'hlittle':
                cosmo_params = p21c.CosmoParams(hlittle=Heff)
            elif name_input1 == 'SIGMA_8':
                cosmo_params = p21c.CosmoParams(SIGMA_8=Heff)
            elif name_input1 == '0Mm':
                cosmo_params = p21c.CosmoParams(OMm=Heff)
            elif name_input1 == '0Mb':
                cosmo_params = p21c.CosmoParams(OMb=Heff)

        initial_conditions = p21c.initial_conditions(
            user_params=user_params,
            cosmo_params=cosmo_params
        )

        data_dict = sa.generate_bias(redshift_range,
                                     initial_conditions,
                                     box_dim,
                                     box_len,
                                     astro_params,
                                     flag_options,
                                     data_dict=data_dict,
                                     nb_bins=20,
                                     comp_zre_PP=True,
                                     logbins=True,
                                     varying_input=name_input1,
                                     varying_in_value=Heff,
                                     comp_ion_hist=True,
                                     comp_bt=False,
                                     return_zre=True)
        print(data_dict)
    return data_dict


# a,b,k = get_params_values( box_len= 200, box_dim= 300, redshift_range = np.linspace(5,18,15))
# heff = np.linspace(20,100,5)
# data_dict = params_changing_run('HII_EFF_FACTOR', heff, redshift_range=np.linspace(5,18,15))
# #print(a,b,k)
# print(data_dict)
# raise ValueError

def parameter_2Dspace_run(name_input1, range1, name_input2, range2, file_name, redshift_range=np.linspace(5, 18, 60),
                          box_dim=143,
                          box_len=143, other_astro_params={"NU_X_THRESH": 500},
                          find_bubble_algorithm=int(2), flag_options={"USE_MASS_DEPENDENT_ZETA": False}, SIGMA_8=0.8,
                          hlittle=0.7, OMm=0.27, OMb=0.045, POWER_INDEX=0.9665,
                          include_zreion=True,
                          comp_brightness_temp=True,
                          ):
    '''
    This function computes the 2dimensional variational space for 2 21cmFAST inputshhyhyuj
    :param name_input1: [string] The name of the first changing astrophysical input (all the possible inputs can be found at : https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html)
    Note! range1 must be in decreasing order for the object file to be like a normal cartesian plane
    :param range1: [list] the range of the desired first input values
    :param name_input2: [string] The name of the second changing astrophysical input (ex: HII_EFF_FACTOR for ionization efficiency (zeta))
    :param range2: [list] the range of the desired second input values
    :param file_name: [string] the name pf the file you want to save the file under (workds with directory to)
    :param redshift_range: [list] the redshift range at which to
    :param box_len: [int] the spatial length of the desired box in Mpc (default is 143 Mpc which is equivalent to 100 Mpc/h)
    :param box_dim: [int] the dimension of the box (number of points per field) (default is 143 for a spatial voxel resolution of (1 Mpc/h)³
    :param other_astro_params: [dict] if you want another astrophysical parameters to stay stable, but under a different value than the default one
    :param flag_options: [dict] a dictionnary of all the wanted non-default flag options parameters on the form { flag_option : value, ...}. This include the use-mass_dependant_zeta function for the usage of astro parameters such as the turnover mass. An extensive list of the usable flag options can be find here : https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :param SIGMA_8: [float] the cosmological value (default 0.8 )
    :param hlittle: [float] the cosmological value (default 0.7)
    :param OMm: [float] the cosmological value (default 0.27)
    :param OMb: [float] the cosmological value (default 0.045)
    :param POWER_INDEX: [float] the cosmological value (default 0.9665 )
    :param find_bubble_algorithm: [int] what method to use when finding the bubbles (default = 2)
    :param include_zreion: [bool] will include z-reion computation and observables if True
    :param comp_brightness_temp: [bool] will compute the brightness temeperature and add it to the model if True
    :return: [2D array] an array containing objects for each of the varying variable run. Each object contains information about 21cmFAST and z-reion. The object structure and attribute can be found on the repo
    '''
    # initialize the storing array
    storing_array = np.empty((10, 10), dtype=object)

    # loop through both varying variabbles
    for count1, Heff in enumerate(tqdm(range1)):
        for count2, Tvir in enumerate(tqdm(range2)):

            # adjustable parameters to look out before running the driver
            use_cache = False  # write True in this line to re-use field and work only on MCMC part (testing and debbuging purposes)
            # radius_thick = 2.  # the radii thickness (will affect the number of bins and therefore points) This is only true for testing purposes of Powerbox
            box_dim = box_dim  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
            box_len = box_len  # int(143) #default value of 300
            user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_dim}
            cosmo_params = p21c.CosmoParams(SIGMA_8=SIGMA_8, hlittle=hlittle, OMm=OMm, OMb=OMb)
            astro_params = other_astro_params | {f'{name_input1}': Heff}
            astro_params = astro_params | {f'{name_input2}': Tvir}
            astro_params = p21c.AstroParams(
                astro_params)  # ","HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir, HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir  #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
            flag_options = p21c.FlagOptions(flag_options)
            p21c.global_params.FIND_BUBBLE_ALGORITHM = find_bubble_algorithm
            initial_conditions = p21c.initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params
            )  # random_seed = Heff

            if os.path.exists('b_mz.npy') and os.path.exists('bmz_errors.npy') and os.path.exists(
                    'kvalues.npy') and use_cache and os.path.exists('zre.npy'):
                # bmz_errors = np.load('bmz_errors.npy')
                # b_mz = np.load('b_mz.npy')
                kvalues = np.load('kvalues.npy')
                z_re_box = np.load('zre.npy')
                density_field = np.load('density.npy')
                overzre, zre_mean = zre.over_zre_field(z_re_box)

            else:

                redshifts = redshift_range
                # comment after variational run
                data_dict = {'Z_re': [], "medians": [], "a16": [], "a50": [], "a84": [],
                             "b16": [], "b50": [], "b84": [], "k16": [], "k50": [], "k84": [], "p16": [], "p50": [],
                             "p84": [],
                             "width50": [], "width90": []}
                cmFast_zre, b_mz, kvalues, data_dict, density_field, cmFAST_hist, zre_PP, den_pp = sa.generate_bias(
                    redshifts, initial_conditions, box_dim, box_len, astro_params, flag_options, data_dict=data_dict,
                    comp_zre_PP=True, logbins=True, comp_ion_hist=True,
                    comp_bt=False)
                # initialize the object
                obj = zrcomp.input_info_field()
                if include_zreion:
                    zre_zreion = zr.apply_zreion(density_field, data_dict['Z_re'][0], data_dict["a50"][0],
                                                 data_dict["k50"][0], box_len, b0=data_dict["b50"][0])
                    over_zre_zreion, zre_mean = np.array(zre.over_zre_field(zre_zreion))
                    # compute zreion power spectrum, and exclude
                    zreion_zre_PP = pbox.get_power(over_zre_zreion, box_len, bins=20, log_bins=True)[0]
                    zreion_zre_PP = zreion_zre_PP[1:]
                    # For testing puroposes with powerbox, this is the line to run homemade powerspectrum
                    # zreion_zre_PP2 = pp.ps_ion_map(over_zre_zreion, 20, box_dim) / (143 ** 3)
                    zreion_hist = pp.reionization_history(redshifts, zre_zreion)
                    obj.set_zreion(zreion_zre_PP, zreion_hist, data_dict["a50"][0], data_dict["b50"][0],
                                   data_dict["k50"][0])
                obj.set_21cmFAST(zre_PP, cmFAST_hist, den_pp, data_dict['Z_re'][0], b_mz)

                # fig, ax = plt.subplots()

                # compute brightness temperature
                bt_ps_21cmFAST = []
                bt_mean_cmFAST = []
                bt_std_cmFAST = []
                bt_ps_zreion = []
                bt_mean_zreion = []
                bt_std_zreion = []
                filenames = []

                # If True, compute the brughtness temperature for z-reion and 21cmFAST
                for z in redshifts:
                    ion2, brightness_temp_cmFAST2 = zrcomp.get_21cm_fields(z, cmFast_zre, density_field)
                    if include_zreion:
                        ion, brightness_temp = zrcomp.get_21cm_fields(z, zre_zreion, density_field)
                        brightness_temp_ps = pbox.get_power(brightness_temp, 143, bins=20, log_bins=True)[0][1:]
                        bt_ps_zreion.append(brightness_temp_ps)
                        bt_mean_zreion.append(np.mean(brightness_temp))
                        bt_std_zreion.append(np.std(brightness_temp))
                    brightness_temp_pscmFAST = pbox.get_power(brightness_temp_cmFAST2, 143, bins=20, log_bins=True)[0][
                                               1:]
                    bt_ps_21cmFAST.append(brightness_temp_pscmFAST)
                    bt_mean_cmFAST.append(np.mean(brightness_temp_cmFAST2))
                    bt_std_cmFAST.append(np.std(brightness_temp_cmFAST2))
                    # plot the bt power_spectrum for a single redshfit
                    # fig, ax = plt.subplots()
                    # plt.scatter(kvalues,brightness_temp_pscmFAST, label ='21cmFAST')
                    # plt.scatter(kvalues, brightness_temp_ps, label = 'z-reion')
                    # #plt.scatter(kvalues, zreion_zre_PP2)
                    # plt.title(f'B_T power spectrum at a redshfit z = {z}')
                    # plt.loglog()
                    # plt.legend()
                    # plt.savefig('./bt_map/bt11_z{}.png'.format(z))
                    # filenames.append('./bt_map/bt11_z{}.png'.format(z))
                    # plt.close()

                obj.cmFASTinfo.add_brightness_temp(bt_ps_21cmFAST, redshifts)
                obj.zreioninfo.add_brightness_temp(bt_ps_zreion, redshifts)

                obj.cmFASTinfo.add_brightness_temp_mean(bt_mean_cmFAST, bt_std_cmFAST)
                obj.zreioninfo.add_brightness_temp_mean(bt_mean_zreion, bt_std_zreion)

                storing_array[count1][count2] = obj

    np.save(file_name, storing_array)
    return


Heff_range = np.linspace(66, 30, 10, endpoint=True)
T_vir_range = np.linspace(4.0, 4.9, 10, endpoint=True)
# parameter_2Dspace_run('HII_EFF_FACTOR', Heff_range, 'ION_Tvir_MIN', T_vir_range, '2D_parameter_space_study_HEFF_25to52_Tvir_40to49_with_mean')
