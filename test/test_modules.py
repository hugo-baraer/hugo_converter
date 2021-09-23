"""
  EoR_research/test/test_modules.py
 
  Author : Hugo Baraer
  Affiliation : McGill University
  Date of creation : 2021-09-23
  
  This module test the functions created in the modules and the driver
  This code is inspired by the test_tool.py file in lab 6 of the PHYS_321 class in 2021 thought by prof. Adrian Liu at Mcgill University
"""

import unittest
import nose.tools as nt
import numpy as np


import z_re_field../project_driver as p_d
import ../z_re_field as zre

class test_modules():

    def setUp(self):
        # Create a random ionization field with 1 or 10 value
        self.dim = 50
        self.rand_field = np.random.randint(1,3,size=(self.dim,self.dim,self.dim))
        self.rand_field[self.rand_field == 1]=-1
        self.rand_field[self.rand_field == 2] = 10

    def tearDown(self):
        pass

    def test_zre(self):
        print(self.rand_array.shape)
        # Test the function to see if it
        # returns an array of the right size
        output_arr = z_re.generate_zre_field(10,5,1,self.dim) #random parameters
        nt.assert_equal(output_arr.shape, (self.dim,self.dim,self.dim))

        # Test the zre_generation function for a known input/output
        test_arr = b = np.array([[[-1,12],[-1,-1]],[[-1,12],[-1,-1]]])
        output_arr = z_re.generate_zre_field(10,0,1,2)
        nt.assert_equal(output_arr[0], 1)
        nt.assert_equal(output_arr[1], 4)

