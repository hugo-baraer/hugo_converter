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

def generate_zre_field(max_z, min_z, z_shift, HII_Dim):
    """
    This function generate a z_re field with coeval cubes at different reionization redshift.
    :param max_z: [float] the maximum redshift of the plot
    :param min_z: [float] the minimum redshift of the plot
    :param z_shift [float] the desired amount different redshift computed for the plot (redshift resolution)
    :param HII_Dim: [float] the minimum redshift of the plot
    :return: a 3D array of the reionization field
    """
    #creating a new cube where reionization vener occured (-1)
    final_cube = np.full((HII_Dim, HII_Dim, HII_Dim), -1)
    for redshift in tqdm(range(min_z, max_z, z_shift)):
        new_cube = p21c.run_coeval(redshift=redshift,user_params={'HII_DIM': HII_Dim, "USE_INTERPOLATION_TABLES":False}).z_re_box
        final_cube[new_cube > -1] = new_cube[new_cube > -1]
    return final_cube

