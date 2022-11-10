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




#storing_array = np.empty((10,10), dtype=object)





# storing_array[0][0] = obj
# np.save('testtt',storing_array)

# print(stoo[0][0].zreioninfo.alpha)
# print(stoo[0][1].zreioninfo.alpha)
# print(stoo[1][0].zreioninfo.alpha)
#print(obj.zreioninfo.P_k_zre)
#varying_input = 'F_star'
#varying_input = 'Turnover_Mass'
#data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}
#ionization_rates = []

#storing_dict = {'21cmFAST':{'P_k_zre':[], 'ion_hist':[], 'P_k_dm':[], 'z_mean':[] ,'b_mz':[]}, 'z_reion_Hugo':{'P_k_zre':[], 'ion_hist':[], 'free_params':[]}}

# z_reion_zre= np.load('zreion_for_Hugo.npz')['zreion']
# reion_hist_zreion = pp.reionization_history(np.linspace(5, 18, 100), z_reion_zre, plot = False)

#print(pp.compute_tau(reion_hist_zreion, redshifts=np.linspace(5,18,100)))

# z_re_box= np.load('zre.npy')
# density_field = np.load('density.npy')
# overzre, zre_mean = zre.over_zre_field(z_re_box)
#
# reion_hist_zreion_0593 = pp.reionization_history(np.linspace(5, 18, 100), zre_zreion, plot=False)
#pp.plot_multiple_ionhist(ion_rates, dictt,varying_input,zreion=reion_hist_zreion, zreion2=reion_hist_zreion_0593)
#pp.plot_variational_range_1dict(dictt, varying_name=varying_input)

# bmzs = []
#
# Heff_range = np.linspace(52, 25,10, endpoint=True)
# T_vir_range = np.linspace(3.8,4.7,10, endpoint=True)
# print(Heff_range, T_vir_range)
#
# print(len(Heff_range))
# Heff_ver, T_vir_hor = np.meshgrid(Heff_range, T_vir_range)
#
# density_small = np.load(f'./density.npy')
#
# user_params = {"HII_DIM": 143, "BOX_LEN": 143, "DIM":143}
# cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
# initial_conditions = p21c.initial_conditions(
#         user_params = user_params,
#         cosmo_params = cosmo_params
#         )
# density_small =  p21c.perturb_field(redshift=8.246021505034825, init_boxes=initial_conditions, write=False).density
# zre_zreion = zr.apply_zreion(density_small,  8.246021505034825, 0.16132113321299762,0.7388640891815643, 100)
#reion_hist_zreion_james = pp.reionization_history(np.linspace(5, 18, 100), zre_zreion, plot=True)
#zrcomp.plot_variational_range_James(dictt, james_alphas, james_k_0, varying_name=varying_input)





#stoo = np.load('Heff25to52_Tvir38to47_varstudy_withzreionbt_withJamesbt_withionhist_corrected.npy', allow_pickle=True)
#wJames_params = np.load('Heff25to52_Tvir38to47_varstudy_cleanbt_withJamesparams.npy', allow_pickle=True)




# objjj = zrcomp.add_James_params(stoo,James_params,Heff_range,T_vir_range)

#np.save('Heff25to52_Tvir38to47_varstudy_cleanbt_withJamesparams', objjj)

#add James_ion_hist

#redshfit_4_bt = stoo[0][0].cmFASTinfo.z_for_bt
# user_params = {"HII_DIM": 143, "BOX_LEN": 143, "DIM":143}
# cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
# initial_conditions = p21c.initial_conditions(
#         user_params = user_params,
#         cosmo_params = cosmo_params
#         )
# objjj = zrcomp.add_James_ion_hist(wJames_params,Heff_range,T_vir_range,initial_conditions)
#
# np.save('Heff25to52_Tvir38to47_varstudy_withzreionbt_withJamesion_hist_nobt', objjj)


#print(len(stoo[0][0].cmFASTinfo.ion_hist))

#print((stoo[5][5].cmFASTinfo.brightnesstemp[15][1:]/(143**3))/stoo[5][5].zreioninfo.brightnesstemp[15][0])
#print(stoo[5][5].zreioninfo.brightnesstemp[15])
# density_small = np.load(f'./density.npy')
# k_values = pbox.get_power(density_small, 143,bins = 20, log_bins = True)[1][1:]
# k_values_James = pbox.get_power(density_small, 100,bins = 20, log_bins = True)[1][1:]
# print(k_values)
# print(k_values)

#cmFAST_TAU = zrcomp.analyze_float_value(stoo, 'James' , 'ion_hist', T_vir_range, Heff_range)
#zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, add_James=True, xaxis=np.linspace(5,18,60))


# zrcomp.analyze_Tau_diff(stoo, 'cmFAST','James','ion_hist', T_vir_range, Heff_range)
# zrcomp.analyze_Tau_diff(stoo, 'zreion','James','ion_hist', T_vir_range, Heff_range)
# zrcomp.analyze_Tau_diff(stoo, 'cmFAST','zreion','ion_hist', T_vir_range, Heff_range)
# print(len(stoo[0][0].cmFASTinfo.P_k_zre))
# print(len(stoo[0][0].zreioninfo.P_k_zre))
#k_values = np.logspace(np.log10(0.08570025), np.log10(7.64144032), 20)



# slice = 0
# zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])

#plot a bunch of stuff
# zrcomp.analyze_float_value(stoo, 'cmFAST' , 'z_mean', T_vir_range, Heff_range)
# zrcomp.analyze_float_value(stoo, 'zreion' , 'alpha', T_vir_range, Heff_range)
# zrcomp.analyze_float_value(stoo, 'zreion' , 'k_0', T_vir_range, Heff_range)
#zrcomp.plot_variational_bias(stoo,'cmFAST','b_mz', T_vir_range, Heff_range,xaxis = k_values, add_zreion=True, log_scale = True)

#for slice in tqdm(range(len(stoo[0][0].cmFASTinfo.z_for_bt)), 'making a reionization movie'):
    #print(getattr(getattr(stoo[0][0], f'Jamesinfo'),'brightnesstemp')[slice]-getattr(getattr(stoo[0][1], f'Jamesinfo'),'brightnesstemp')[slice])
#print(getattr(getattr(stoo[0][0], f'Jamesinfo'), 'brightnesstemp')[20] -
      #getattr(getattr(stoo[9][9], f'Jamesinfo'), 'brightnesstemp')[20])


# for count1, Heff in enumerate(tqdm(Heff_range)):
#     for count2, Tvir in enumerate(tqdm(T_vir_range)):
#         print(getattr(getattr(stoo[count1][count2], f'Jamesinfo'),'brightnesstemp')[20]-getattr(getattr(stoo[count1+1][count2], f'Jamesinfo'),'brightnesstemp')[20])
#
#         continue

#make a brightness temperature movie
def make_bt_movie(stoo, k_values, add_zreion= True, add_James = True):
    '''
    This function makes a brightness temperature movie
    :param stoo: the object containing all the brightness temperature information
    :param kvalues: the values of k of the power_spectrum computations
    :param add_zreion: (default True) add z-reion info if True
    :param add_James: (default True) add James' algorithm info if True
    :return:
    '''
    filenames = []
    for slice in tqdm(range(len(stoo[0][0].cmFASTinfo.z_for_bt)), 'making a reionization movie'):
        filenames = zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', stoo[0][0].cmFASTinfo.z_for_bt[slice],slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=add_zreion,savefig = True, filenames = filenames, add_James=add_James, xaxis_James = k_values_James)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('bt_fct_redshift_cleanbt_final_sameylims.gif', images)
    return


#zrcomp.plot_multiple_ion_hist(stoo,'zreion','P_k_zre', T_vir_range, Heff_range)
# zrcomp.plot_variational_PS(stoo,'cmFAST','P_k_zre', T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, plot_diff=True, xaxis=np.linspace(5,18,60))
# zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, xaxis=np.linspace(5,18,60))
#p_k_zre, kbins_zre = pbox.get_power(stoo[0][0].cmFASTinfo.P_k_zre,143)
#
#



#add z-reion_bt or James_bt
# redshfit_4_bt = stoo[0][0].cmFASTinfo.z_for_bt
# user_params = {"HII_DIM": 143, "BOX_LEN": 143, "DIM":143}
# cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
# initial_conditions = p21c.initial_conditions(
#         user_params = user_params,
#         cosmo_params = cosmo_params
#         )
# objjj = zrcomp.add_James_bt(stoo,redshfit_4_bt,Heff_range,T_vir_range,initial_conditions)

#np.save('Heff25to52_Tvir38to47_varstudy_withzreionbt_withJamesbt_withionhist_corrected', objjj)

#raise ValueError

#zre_mean = [6.8,7.0,7.2,7.4,7.6,7.8,8.0]
#random_seed = [12345,54321,23451,34512,45123]

#random_seed = random_seed[0]

def compute_several_21cmFASt_fields(random_seed, zre_mean, astro_params ={"NU_X_THRESH": 500}, find_bubble_algorithm = 2):
    '''
    This function generates fields and save fields at various redshift and random seeds.
    :param random_seed: [list] all the desired random seed used
    :param zre_mean: [list] the desired computation redshifts
    :param astro_params: the astro_params you want the fields to have (cosmological parameters can be changed by hand.
    :param find_bubble_algorithm: [int] The find bubble algorithm method use (default 2)
    :return: saved fields
    '''
    for count1, Heff in enumerate(tqdm(random_seed)):
        for count2, Tvir in enumerate(tqdm(zre_mean)):

            # adjustable parameters to look out before running the driver
            # change the dimension of the box and see effect9
            # density_small = np.load(f'./fields_final/method_2_density_field_z_{Tvir}_random_seed_{Heff}.npy')
            # xH_small = np.load(f'./fields_final/method_2_xH_z_{Tvir}_random_seed_{Heff}.npy')
            # #b_t = np.load(f'./method_2_brightness_temp_z_{Tvir}_random_seed_{Heff}_dim200_len300Mpc.npy')
            # # density = np.load(f'./fields_final/method_1_density_field_z_{Tvir}_random_seed_{Heff}.npy')
            # xH = np.load(f'./fields_300MPC_21cmFAST_final/method_2_xH_z_{Tvir}_random_seed_{Heff}_dim200_len300Mpc.npy')
            # density = np.load(f'./fields_300MPC_21cmFAST_final/method_2_density_field_z_{Tvir}_random_seed_{Heff}_dim200_len300Mpc.npy')
            # xH = np.load(f'./fields_for_Adrian/method_1_xH_z_{Tvir}_random_seed_{Heff}.npy')
            # xH = np.load(f'./corrected_field/method_1_xH_z_{Tvir}_random_seed_{Heff}_dim200.npy')
            # brightness_temp = p21c.brightness_temperature(ionized_box=ionized_box, perturbed_field=perturbed_field)
            # np.save(f'method_1_brightness_temp_z_{zre_mean}_random_seed_{random_seed}', brightness_temp.brightness_temp)

            use_cache = False  # uncomment this line to re-use field and work only on MCMC part
            box_dim = 143  # the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
            radius_thick = 2.  # the radii thickness (will affect the number of bins and therefore points)
            box_len = 143  # int(143) #default value of 300
            user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM": box_dim}
            cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045)
            astro_params = p21c.AstroParams(astro_params)  # ","HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir, HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir  #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
            flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": False})
            # add astro_params
            compare_with_james = False
            p21c.global_params.FIND_BUBBLE_ALGORITHM = find_bubble_algorithm
            initial_conditions = p21c.initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                random_seed = Heff
            )


            zrcomp.compute_field_Adrian(Tvir, initial_conditions, astro_params, flag_options, random_seed=Heff)


#def compute21cmFAST_zre_field():
def get_params_values(box_len = 143, box_dim = 143, include_confidencerange = False, redshift_range = np.linspace(5,18,60), nb_bins = 20, density_field = None, zre_field = None, plot_best_fit = False, plot_corner = False, return_zre_field = False, return_density = False, return_power_spectrum = False, astro_params= {"HII_EFF_FACTOR": 30.0}, flag_options = {"USE_MASS_DEPENDENT_ZETA": False}, SIGMA_8= 0.8, hlittle=0.7, OMm=0.27, OMb=0.045, POWER_INDEX = 0.9665, find_bubble_algorithm = 2):
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
    astro_params = p21c.AstroParams(astro_params)  # ","HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir, HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir  #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
    flag_options = p21c.FlagOptions(flag_options)
    #p21c.global_params.FIND_BUBBLE_ALGORITHM = find_bubble_algorithm
    #test
    initial_conditions = p21c.initial_conditions(user_params=user_params,cosmo_params=cosmo_params)  # random_seed = Heff

    # comment after variational run
    data_dict = {'Z_re': [], "medians": [], "a16": [], "a50": [], "a84": [],
                 "b16": [], "b50": [], "b84": [], "k16": [], "k50": [], "k84": [], "p16": [], "p50": [], "p84": [],
                 "width50": [], "width90": []}

    if return_zre_field and return_density:
        data_dict, density_field, z_re_field = sa.generate_bias(redshift_range, initial_conditions, box_dim, box_len, astro_params,
                                                    flag_options, density_field=density_field, z_re_box=zre_field,
                                                    data_dict=data_dict, nb_bins=20, plot_best_fit=plot_best_fit,
                                                    plot_corner=plot_corner, comp_zre_PP=False, logbins=True,
                                                    comp_ion_hist=False, comp_bt=False, return_zre=return_zre_field, return_density= return_density)

    elif return_zre_field or return_density:
        data_dict, single_field = sa.generate_bias(redshift_range, initial_conditions, box_dim, box_len,astro_params,
                                                    flag_options, density_field=density_field, z_re_box=zre_field,
                                                    data_dict=data_dict, nb_bins=20, plot_best_fit=plot_best_fit,
                                                    plot_corner=plot_corner, comp_zre_PP=False, logbins=True,
                                                    comp_ion_hist=False, comp_bt=False, return_zre= return_zre_field, return_density= return_density)
    else:
        data_dict= sa.generate_bias(redshift_range, initial_conditions, box_dim, box_len,astro_params,
                                                    flag_options, density_field=density_field, z_re_box=zre_field,
                                                    data_dict=data_dict, nb_bins=20, plot_best_fit=plot_best_fit,
                                                    plot_corner=plot_corner, comp_zre_PP=False, logbins=True,
                                                    comp_ion_hist=False, comp_bt=False, return_zre= return_zre_field, return_density= return_density)


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


#a, b, k, density_field = get_params_values(include_confidencerange=True,  redshift_range=np.linspace(5, 18, 15), return_density=True)

#print(a, b, k)
#print('coucou')

def params_changing_run(name_input1, range1, redshift_range= np.linspace(5,18,60), box_dim=143, box_len=143,is_astro=True,other_astro_params={"NU_X_THRESH": 500},find_bubble_algorithm=2, flag_options={"USE_MASS_DEPENDENT_ZETA": False}, plot_best_fit = False, plot_corner = False):
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
            cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
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
                                    varying_in_value= Heff,
                                    comp_ion_hist=True,
                                    comp_bt=False,
                                    return_zre=True)
        print(data_dict)
    return data_dict

a,b,k = get_params_values( box_len= 200, box_dim= 300, redshift_range = np.linspace(5,18,15))
# heff = np.linspace(20,100,5)
# data_dict = params_changing_run('HII_EFF_FACTOR', heff, redshift_range=np.linspace(5,18,15))
# #print(a,b,k)
# print(data_dict)
# raise ValueError

def parameter_2Dspace_run(name_input1, range1, name_input2, range2, redshift_range = np.linspace(5,18,60), box_dim = 143, box_len = 143, is_astro1 = True, is_astro2 = True, other_astro_params = { "NU_X_THRESH":500}, find_bubble_algorithm = 2, flag_options = {"USE_MASS_DEPENDENT_ZETA": False}, include_zreion = True ):
    '''
    This function computes the 2dimensional variational space for 2 21cmFAST inputshhyhyuj
    :param name_input1:
    :param range1:
    :param name_input2:
    :param range2:
    :param redshift_range:
    :param box_dim:
    :param box_len:
    :param is_astro1:
    :param is_astro2:
    :param other_astro_params:
    :param find_bubble_algorithm:
    :param flag_options:
    :param include_zreion:
    :return:
    '''

    for count1, Heff in enumerate(tqdm(range1)):
        for count2, Tvir in enumerate(tqdm(range2)):

            #adjustable parameters to look out before running the driver
            #change the dimension of the box and see effect9
            # density_small = np.load(f'./fields_final/method_2_density_field_z_{Tvir}_random_seed_{Heff}.npy')
            # xH_small = np.load(f'./fields_final/method_2_xH_z_{Tvir}_random_seed_{Heff}.npy')
            # #b_t = np.load(f'./method_2_brightness_temp_z_{Tvir}_random_seed_{Heff}_dim200_len300Mpc.npy')
            # # density = np.load(f'./fields_final/method_1_density_field_z_{Tvir}_random_seed_{Heff}.npy')
            # xH = np.load(f'./fields_300MPC_21cmFAST_final/method_2_xH_z_{Tvir}_random_seed_{Heff}_dim200_len300Mpc.npy')
            # density = np.load(f'./fields_300MPC_21cmFAST_final/method_2_density_field_z_{Tvir}_random_seed_{Heff}_dim200_len300Mpc.npy')
            #xH = np.load(f'./fields_for_Adrian/method_1_xH_z_{Tvir}_random_seed_{Heff}.npy')
            #xH = np.load(f'./corrected_field/method_1_xH_z_{Tvir}_random_seed_{Heff}_dim200.npy')
            # brightness_temp = p21c.brightness_temperature(ionized_box=ionized_box, perturbed_field=perturbed_field)
            # np.save(f'method_1_brightness_temp_z_{zre_mean}_random_seed_{random_seed}', brightness_temp.brightness_temp)

            use_cache = False # uncomment this line to re-use field and work only on MCMC part
            box_dim = 143 #the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
            radius_thick = 2. #the radii thickness (will affect the number of bins and therefore points)
            box_len = 143 #int(143) #default value of 300
            user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM":box_dim}
            cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
            astro_params = p21c.AstroParams({ "NU_X_THRESH":500}) #","HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir, HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir  #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
            flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": False})
            #add astro_params
            compare_with_james = False
            p21c.global_params.FIND_BUBBLE_ALGORITHM = 2
            initial_conditions = p21c.initial_conditions(
            user_params = user_params,
            cosmo_params = cosmo_params
            ) #random_seed = Heff
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

                zrcomp.compute_field_Adrian(Tvir,initial_conditions,astro_params,flag_options,random_seed=Heff)
                continue

                redshifts = np.linspace(5,18,60)

                #comment after variational run
                data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16": [], "a50": [], "a84": [],
                             "b16": [], "b50": [], "b84": [], "k16": [], "k50": [], "k84": [], "p16": [], "p50": [], "p84": [],
                             "width50": [], "width90": []}
                varying_in_value = 1.
                cmFast_zre, b_mz, kvalues, data_dict, density_field, cmFAST_hist, zre_PP, den_pp = sa.generate_bias(redshifts, initial_conditions, box_dim, astro_params, flag_options, varying_input,
                                  varying_in_value, data_dict=data_dict,comp_zre_PP=True, logbins=True, comp_ion_hist=True ,comp_bt=False, return_zre= True)


                obj = zrcomp.input_info_field()
                zre_zreion = zr.apply_zreion(density_field, data_dict['Z_re'][0], data_dict["a50"][0],data_dict["k50"][0], 143, b0 = data_dict["b50"][0])
                over_zre_zreion, zre_mean = np.array(zre.over_zre_field(zre_zreion))

                zreion_zre_PP = pbox.get_power(over_zre_zreion, 143, bins=20, log_bins=True)[0]
                zreion_zre_PP= zreion_zre_PP[1:]

                zreion_zre_PP2 = pp.ps_ion_map(over_zre_zreion, 20, box_dim) /(143**3)
                zreion_hist = pp.reionization_history(redshifts,zre_zreion)
                obj.set_zreion(zreion_zre_PP, zreion_hist, data_dict["a50"][0],data_dict["b50"][0],data_dict["k50"][0])
                obj.set_21cmFAST(zre_PP,cmFAST_hist,den_pp,data_dict['Z_re'][0],b_mz)


                #fig, ax = plt.subplots()

                #compute brightness temperature

                bt_ps_21cmFAST = []
                bt_ps_zreion = []
                filenames = []
                for z in redshifts:
                    ion2, brightness_temp_cmFAST2 = zrcomp.get_21cm_fields(z, cmFast_zre, perturbed_field.density)
                    ion, brightness_temp = zrcomp.get_21cm_fields(z, zre_zreion, perturbed_field.density)
                    brightness_temp_ps = pbox.get_power(brightness_temp, 143, bins = 20, log_bins = True)[0][1:]
                    brightness_temp_pscmFAST = pbox.get_power(brightness_temp_cmFAST2, 143, bins = 20, log_bins = True)[0][1:]
                    bt_ps_21cmFAST.append(brightness_temp_pscmFAST)
                    bt_ps_zreion.append(brightness_temp_ps)

                    #plot the bt for a single redshfit
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

                obj.cmFASTinfo.add_brightness_temp(bt_ps_21cmFAST,redshifts)
                obj.zreioninfo.add_brightness_temp(bt_ps_zreion,redshifts)

                # images = []
                # for filename in filenames:
                #     images.append(imageio.imread(filename))
                # imageio.mimsave('bt_fct_redshift_singleplot.gif', images)


                #ioniozation histories
                # fig, ax = plt.subplots()
                # plt.scatter(redshifts,cmFAST_hist, label ='21cmFAST')
                # plt.scatter(redshifts, zreion_hist, label = 'z-reion')
                # plt.loglog()
                # #plt.scatter(kvalues, zreion_zre_PP2)
                #
                # plt.legend()
                # plt.show()


                #test b_t stuff
                #ion, brightness_temp = get_21cm_fields(8.0, zre_zreion, perturbed_field.density)




                #print(obj.cmFASTinfo.brightnesstemp)
                storing_array[count1][count2] = obj
                #bmzs.append(b_mz.tolist())
                #print(ionization_rates)


            #y_plot_fit = sa.lin_bias(kvalues, 0.564,0.593,0.185)
    #np.save('Heff25to52_Tvir38to47_varstudy_cleanbt', storing_array)
    #print(storing_array)
    # print("bmzs = ", bmzs)
    # print("dictt = ", data_dict)
    # print("ion_rates =" , ionization_rates)