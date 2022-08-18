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




storing_array = np.empty((10,10), dtype=object)

# storing_array[0][0] = obj
# np.save('testtt',storing_array)

# print(stoo[0][0].zreioninfo.alpha)
# print(stoo[0][1].zreioninfo.alpha)
# print(stoo[1][0].zreioninfo.alpha)
#print(obj.zreioninfo.P_k_zre)
varying_input = 'F_star'
#varying_input = 'Turnover_Mass'
data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[], "width50":[],"width90":[]}
ionization_rates = []

storing_dict = {'21cmFAST':{'P_k_zre':[], 'ion_hist':[], 'P_k_dm':[], 'z_mean':[] ,'b_mz':[]}, 'z_reion_Hugo':{'P_k_zre':[], 'ion_hist':[], 'free_params':[]}}

z_reion_zre= np.load('zreion_for_Hugo.npz')['zreion']
reion_hist_zreion = pp.reionization_history(np.linspace(5, 15, 100), z_reion_zre, plot = False)

print(pp.compute_tau(reion_hist_zreion, redshifts=np.linspace(5,15,100)))

# z_re_box= np.load('zre.npy')
# density_field = np.load('density.npy')
# overzre, zre_mean = zre.over_zre_field(z_re_box)
#
# reion_hist_zreion_0593 = pp.reionization_history(np.linspace(5, 15, 100), zre_zreion, plot=False)
#pp.plot_multiple_ionhist(ion_rates, dictt,varying_input,zreion=reion_hist_zreion, zreion2=reion_hist_zreion_0593)
#pp.plot_variational_range_1dict(dictt, varying_name=varying_input)

bmzs = []

Heff_range = np.linspace(52, 25,10, endpoint=True)
T_vir_range = np.linspace(3.8,4.7,10, endpoint=True)
print(Heff_range, T_vir_range)

Heff_ver, T_vir_hor = np.meshgrid(Heff_range, T_vir_range)



stoo = np.load('Heff25to52_Tvir38to47_varstudy.npy', allow_pickle=True)
#print(len(stoo[0][0].cmFASTinfo.ion_hist))

#print((stoo[5][5].cmFASTinfo.brightnesstemp[15][1:]/(143**3))/stoo[5][5].zreioninfo.brightnesstemp[15][0])
#print(stoo[5][5].zreioninfo.brightnesstemp[15])
k_values = pbox.get_power(np.ones((143,143,143)), 100,bins = 20, log_bins = True)[1]
print(k_values)
#cmFAST_TAU = zrcomp.analyze_float_value(stoo, 'zreion' , 'ion_hist', T_vir_range, Heff_range)
#zrcomp.analyze_Tau_diff(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range)
# print(len(stoo[0][0].cmFASTinfo.P_k_zre))
# print(len(stoo[0][0].zreioninfo.P_k_zre))
#k_values = np.logspace(np.log10(0.08570025), np.log10(7.64144032), 20)



# slice = 0
# zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])
#
# slice = 7
# zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])
#
# slice = 14
# zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])
#
# slice = 21
# zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])
#
# slice = 28
# zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', slice,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
# print(stoo[0][0].cmFASTinfo.z_for_bt[slice])

#plot a bunch of stuff
# zrcomp.analyze_float_value(stoo, 'cmFAST' , 'z_mean', T_vir_range, Heff_range)
# zrcomp.analyze_float_value(stoo, 'zreion' , 'alpha', T_vir_range, Heff_range)
# zrcomp.analyze_float_value(stoo, 'zreion' , 'k_0', T_vir_range, Heff_range)
zrcomp.plot_variational_bias(stoo,'cmFAST','b_mz', T_vir_range, Heff_range,xaxis = k_values, add_zreion=True, log_scale = True)
#zrcomp.plot_variational_bright_temp(stoo, 'cmFAST', 'brightnesstemp', 15,T_vir_range, Heff_range,xaxis = k_values, add_zreion=True)
#zrcomp.plot_multiple_ion_hist(stoo,'zreion','P_k_zre', T_vir_range, Heff_range)
zrcomp.plot_variational_PS(stoo,'cmFAST','P_k_zre', T_vir_range, Heff_range,xaxis = k_values[1:], add_zreion=True, delta2= True)
zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, plot_diff=True, xaxis=np.linspace(5,18,60))
zrcomp.plot_variational_ion_hist(stoo, 'cmFAST', 'ion_hist', T_vir_range, Heff_range, add_zreion = True, xaxis=np.linspace(5,18,60))
p_k_zre, kbins_zre = pbox.get_power(stoo[0][0].cmFASTinfo.P_k_zre, 100)
#
#
# #add z-reion_bt
# redshfit_4_bt = stoo[0][0].cmFASTinfo.z_for_bt
# user_params = {"HII_DIM": 143, "BOX_LEN": 143, "DIM":143}
# cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
# initial_conditions = p21c.initial_conditions(
#         user_params = user_params,
#         cosmo_params = cosmo_params
#         )
# objjj = zrcomp.add_zreion_bt(stoo,redshfit_4_bt,Heff_range,T_vir_range,initial_conditions)

#np.save('Heff15to45_Tvir40to49_varstudy_withbt', objjj)

#raise ValueError


zre_mean = [6.8,7.0,7.2,7.4,7.6,7.8,8.0]
random_seed = [12345,54321,23451,34512,45123]

#random_seed = random_seed[0]

for count1, Heff in enumerate(tqdm(Heff_range)):
    for count2, Tvir in enumerate(tqdm(T_vir_range)):
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
        astro_params = p21c.AstroParams({ "NU_X_THRESH":500, "HII_EFF_FACTOR": Heff, "ION_Tvir_MIN":Tvir}) #"HII_EFF_FACTOR":Heff = 44 #for adrian optimization, "M_TURN" : Heff "M_TURN":10, "F_STAR10": Heff, "F_ESC10":-0.08
        flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": False})
        #add astro_params
        compare_with_james = False
        #p21c.global_params.FIND_BUBBLE_ALGORITHM = 1
        initial_conditions = p21c.initial_conditions(
        user_params = user_params,
        cosmo_params = cosmo_params
        ) #random_seed = Heff
        dummy_count = 0

        # zre_range = np.linspace(8,10,2)
        # # z_re_box, b_temp_ps, z_4_bt = zre.generate_zre_field(zre_range, initial_conditions, box_dim, astro_params,
        # #                                                      flag_options,
        # #                                                      comP_ionization_rate=False, comp_brightness_temp=True)
        # perturbed_field = p21c.perturb_field(redshift=8.1, init_boxes=initial_conditions, write=False)
        # zre_zreion = zr.apply_zreion(perturbed_field.density,
        #                            8.1,0.6,0.4,100,b0 = 1.0)
        # zre.plot_zre_slice(zre_zreion, 143, 143)
        # print(pbox.get_power(zre_zreion,100, bins = 20, log_bins = True))
        # print(pbox.get_power(zre_zreion, 143, bins=20, log_bins=True))
        # print('coucou')
        # zre_zreion1 = zr.apply_zreion(perturbed_field.density,
        #                            8.1,0.6,0.4,143,b0 = 1.0)
        # zre.plot_zre_slice(zre_zreion1, 143, 143)
        #
        # fig, ax = plt.subplots()
        # plt.scatter(pbox.get_power(perturbed_field.density,100, bins = 20, log_bins = True)[1], pbox.get_power(zre_zreion,100, bins = 20, log_bins = True)[0], label='100')
        # plt.scatter(pbox.get_power(zre_zreion1,100, bins = 20, log_bins = True)[1], pbox.get_power(zre_zreion1,100, bins = 20, log_bins = True)[0], label='143')
        # plt.legend()
        # plt.show()
        # break
        #aaa = p21c.ionize_box(redshift=Tvir, init_boxes = initial_conditions, astro_params = astro_params, flag_options = flag_options, write=False)
        # box = aaa.xH_box
        #print(aaa.xH_box)
        #print(aaa.z_re_box)
        #print(aaa.xH_box + aaa.z_re_box)
        # zre.plot_zre_slice(density_small)
        # zre.plot_zre_slice(density, resolution=200, size = 300)
        # zre.plot_zre_slice(b_t, resolution=200, size = 300)
        # zre.plot_zre_slice(density)
        # #zre.plot_zre_slice(aaa.z_re_box)
        # nuetral_num = np.sum(xH.flatten() > 0)
        # print( nuetral_num / len(xH.flatten()))
        # # # print(xH.sum() / (143 ** 3))
        # # # print(aaa.xH_box.sum() / 143 ** 3)
        #continueb_mz = b_mz1[1:]

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

            # zrcomp.compute_field_Adrian(Tvir,initial_conditions,astro_params,flag_options,random_seed=Heff)
            # continue

            redshifts = np.linspace(5,18,60)

            #comment after variational run
            data_dict = {'Z_re': [], '{}'.format(varying_input): [], "medians": [], "a16": [], "a50": [], "a84": [],
                         "b16": [], "b50": [], "b84": [], "k16": [], "k50": [], "k84": [], "p16": [], "p50": [], "p84": [],
                         "width50": [], "width90": []}
            varying_in_value = 1.
            b_mz, kvalues, data_dict, density_field, cmFAST_hist, zre_PP, den_pp, b_t_ps, z_4_bt = sa.generate_bias(redshifts, initial_conditions, box_dim, astro_params, flag_options, varying_input,
                              varying_in_value, data_dict=data_dict,comp_zre_PP=True)
            obj = zrcomp.input_info_field()
            #kvalues = pbox.get_power(density_field, 100,bins = 20, log_bins = True)[1]
            zre_zreion = zr.apply_zreion(density_field, data_dict['Z_re'][0], data_dict["a50"][0],data_dict["k50"][0], 100, b0 = data_dict["b50"][0])
            zreion_zre_PP = pbox.get_power(zre_zreion, 100, bins=20, log_bins=True)
            # zreion_zre_PP = pp.ps_ion_map(zre_zreion, 20, box_dim)
            zreion_hist = pp.reionization_history(redshifts,zre_zreion)
            obj.set_zreion(zreion_zre_PP, zreion_hist, data_dict["a50"][0],data_dict["b50"][0],data_dict["k50"][0])
            obj.set_21cmFAST(zre_PP,cmFAST_hist,den_pp,data_dict['Z_re'][0],b_mz)
            obj.cmFASTinfo.add_brightness_temp(b_t_ps,z_4_bt)
            storing_array[count1][count2]=obj
            print(obj.cmFASTinfo.brightnesstemp)
            #bmzs.append(b_mz.tolist())
            #print(ionization_rates)


        #y_plot_fit = sa.lin_bias(kvalues, 0.564,0.593,0.185)
#np.save('Heff25to52_Tvir38to47_varstudy', storing_array)
#print(storing_array)
print("bmzs = ", bmzs)
print("dictt = ", data_dict)
print("ion_rates =" , ionization_rates)