"""
  EoR_research/project_driver.py
 
  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-20
  
  This module computes the over_redshift z_re field with coeval cubes at different reionization redshift.
  
"""
import numpy as np
import py21cmfast as p21c
from tqdm import tqdm
from py21cmfast import plotting
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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


def generate_zre_field(max_z, min_z, z_shift,initial_conditions,box_dim, astro_params, flag_options):
    """
    This function generate a z_re field with coeval cubes at different reionization redshift.
    :param max_z: [float] the maximum redshift of the plot
    :param min_z: [float] the minimum redshift of the plot
    :param z_shift [float] the desired amount different redshift computed for the plot (redshift resolution)
    :param HII_Dim: [float] the minimum redshift of the plot
    :return: a 3D array of the reionization field
    """
    #creating a new cube where reionization vener occured (-1)
    final_cube = np.full((box_dim, box_dim, box_dim), -1)
    for redshift in tqdm(np.arange(min_z, max_z, z_shift), 'computing the redshift of reionization'):
        new_cube = p21c.ionize_box(redshift=redshift, init_boxes = initial_conditions, astro_params = astro_params, flag_options = flag_options, write=False).z_re_box
        final_cube[new_cube > -1] = new_cube[new_cube > -1]
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

def over_zre_field(zre_field):
    """
    This function generate the over z_re field with the original z_re field. from the equation defined in Battaglia et al.
    :param zre_field: [arr] 3D array of the reionization redshift field
    :return: a 3D array of the reionization field
    """
    zre_mean = np.mean(zre_field)
    return over_zre_equation(zre_field,zre_mean), zre_mean

def generate_gaussian_field(dim):
    '''
    This function creates a 3d Gaussian field to test np.FFT. This functions is inspired by :
    https://stackoverflow.com/questions/25720600/generating-3d-gaussian-distribution-in-python
    :param dim: the dimension of the desired Gaussian field
    :type dim: int
    :return: the 3d Gaussian field
    :rtype: 3D array
    '''
    x, y, z = np.mgrid[-1.0:1.0:int(dim)*1j, -1.0:1.0:int(dim)*1j, -1.0:1.0:int(dim)*1j]
    # Need an (N, 2) array of (x, y) pairs.
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([.050, .050, .050])
    covariance = np.diag(sigma ** 2)
    zi = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)
    zi2 = np.reshape(zi, (x.shape))
    # Reshape back to a (30, 30) grid.
    return zi2, mu[0], sigma[0]

