"""
  EoR_research/project_driver.py
 
  Author : Hugo Baraer
  Affiliation : McGill University
  Supervision by : Prof. Adrian Liu
  Date of creation : 2021-09-20
  
  This module computes the over_redshift z_re field with coeval cubes at different reionization redshift.
  
"""
import numpy as np
import py21cmfast as p21c
from tqdm import tqdm
from py21cmfast import plotting
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import plot_params as pp
import powerbox as pbox
def generate_quick_zre_field(max_z, min_z, z_shift, initial_conditions):
    """
    This function generate a z_re field with coeval cubes at different reionization redshift.
    :param max_z: [float] the maximum redshift of the plot
    :param min_z: [float] the minimum redshift of the plot
    :param z_shift [float] the desired amount different redshift computed for the plot (redshift resolution)
    :param HII_Dim: [float] the minimum redshift of the plot
    :return: a 3D array of the reionization field
    """
    #creating a new cube where reionization vener occured (-1)
    redshifts = np.linspace(min_z,max_z, max_z)
    print(redshifts)
    """
    perturbed_field=p21c.perturb_field(
            redshift=z,
            init_boxes=initial_conditions,
            zprime_step_factor = 1.5,
            z_heat_max = max_z
        )
    print('-------------------------------------------')
    print('Computing the redshift of reionization field')
    print('-------------------------------------------')
    ionized_field = p21c.ionize_box(
        perturbed_field=perturbed_field,
    )
    plotting.coeval_sliceplot(ionized_field, "z_re_box");
    """
    zre_fields = p21c.ionize_box(redshift=8, init_boxes = initial_conditions)
    #zre_fields = p21c.run_coeval(redshift=redshifts, init_box=initial_conditions)
    # final_cube = np.full((HII_Dim, HII_Dim, HII_Dim), -1)
    # for redshift in tqdm(np.arange(min_z, max_z, z_shift), 'computing the redshift of reionization'):
    #     new_cube = p21c.run_coeval(redshift=redshift,user_params={'HII_DIM': HII_Dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES":False}).z_re_box
    #     final_cube[new_cube > -1] = new_cube[new_cube > -1]
    return


def generate_zre_field(zre_range,initial_conditions,box_dim, astro_params, flag_options, comP_ionization_rate = False, comp_brightness_temp = False, i = 0):
    """
    This function generate a z_re field with coeval cubes at different reionization redshift.
    :param zre_range: the desired redhsift to compute the redshift of reionization field on
    :type zre_range: list
    :param initial_conditions: the initial p21c conditions
    :type initial_conditions: object
    :param box_dim: the dimension (#of pixels) of the computed field
    :type box_dim: float
    :param astro_params: the defined astro_parameters (ex Ionization efficiency and turnover mass)
    :type astro_params: P21C object
    :param flag_options: the flag options of 21cmFAST (like no caching and use the turnover mass factor)
    :type flag_options:  P21c object
    :return: a 3D array of the reionization field
    :rtype:
    """

    #creating a new cube where reionization vener occured (-1)
    final_cube = np.full((box_dim, box_dim, box_dim), -1, dtype = float)
    #if comP_ionization_rate : ionization_rate = []
    if comp_brightness_temp:
        b_temp_ps = []
        redshifts4bright = []
    for redshift in tqdm(zre_range, 'computing the redshift of reionization',position=0, leave=True):
        #print(redshift)
        i += 1
        ionize_box = p21c.ionize_box(redshift=redshift, init_boxes = initial_conditions, astro_params = astro_params, flag_options = flag_options, write=False)
        new_cube = ionize_box.z_re_box
        #if comP_ionization_rate: ionization_rate.append((new_cube > -1).sum()/ box_dim**3)
        if comp_brightness_temp and i%2 == 0 :
            perturbed_field = p21c.perturb_field(redshift=redshift, init_boxes=initial_conditions, write=False)
            brightness_temp = p21c.brightness_temperature(ionized_box=ionize_box, perturbed_field=perturbed_field, write=False).brightness_temp
            brightness_temp_ps = pbox.get_power(brightness_temp, 100, bins = 20, log_bins=True)
            b_temp_ps.append(brightness_temp_ps[0])
            redshifts4bright.append(redshift)
        final_cube[new_cube > -1] = redshift
    if comp_brightness_temp:
        return final_cube, b_temp_ps, redshifts4bright
    else:
        return final_cube

def over_zre_equation(zre_x,zre_mean):
    '''

    :param zre_x: the reionization redshift field
    :type zre_x: 3D array
    :param zre_mean: the mean redshift of reionization
    :type zre_mean:  float
    :return: over-redshift field
    :rtype: 3D array
    '''
    return((1+zre_x)-(1+zre_mean))/(1+zre_mean)

def over_p_equation(p,p_mean):
    '''

    :param p: the density field
    :type p: 3D array
    :param p_mean: the mean of the density field
    :type p_mean:  float
    :return: over-density field
    :rtype: 3D array
    '''
    return(p -p_mean)/(p_mean)

def over_p_field(p_field):
    """
    This function generate the over z_re field with the original z_re field. from the equation defined in Battaglia et al.
    :param p_field: [arr] 3D array of the density field
    :return: a 3D array of the reionization field
    """
    p_mean = np.mean(p_field)
    return over_p_equation(p_field,p_mean), p_mean

def over_zre_field(zre_field):
    """
    This function generate the over z_re field with the original z_re field. from the equation defined in Battaglia et al.
    :param zre_field: [arr] 3D array of the reionization redshift field
    :return: a 3D array of the reionization field
    """
    zre_mean = np.mean(zre_field.flatten())
    return over_zre_equation(zre_field,zre_mean), zre_mean

def plot_zre_slice(field,  resolution = 143, size = 143):
    '''
    This modules plot a slice of the redshift of reionization for a given redshift of reionization 3D field. I also converts Mpc/h to Mpc units while plotting
    :param field: (3D array) redshift of reionization field
    :param resolution: (int) the dimension of the cub
    :return:
    '''
    if resolution % 2:
        position_vec = np.linspace(-int((size*(100/143))//2)-1, int(size*(100/143)//2), resolution)
    else:
        position_vec = np.linspace(-int((size*(100/143))//2), int(size*(100/143)//2), resolution)
    X, Y = np.meshgrid(position_vec, position_vec)
    fig, ax = plt.subplots()
    plt.contourf(X,Y,field[int(resolution//2)])
    plt.colorbar()
    ax.set_xlabel(r'[Mpc h⁻¹]')
    ax.set_ylabel(r'[Mpc h⁻¹]')
    plt.title(
        r'slice of a the over-redshift of reionization at the center with a pixel resolution of {} Mpc h⁻¹'.format('1'))
    plt.show()

def plot_zre_hist(field, nb_bins = 100):
    '''
    This module plots the histograms for redshift reionization field
    :param field:  (3D array) redshift of reionization field
    :return:
    '''
    fig, ax = plt.subplots()
    plt.hist(field.flatten(), bins=nb_bins, density=True)
    plt.show()


