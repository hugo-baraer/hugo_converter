# Hugo converter documentation

This folder contains the necessary documentation to understand the incredible hugo converter. In the subfolder presentation_slides, you'll find the slides for the three presentations given at the Cosmic Dawn Group at McGill University. In the final_report folder, you'll find the final report for two research classes done during the first and second semesters of work. Finally,  you'll find presentation.ipynb, a jupyter notebook showing the main functions, examples of their usage, their options, and exactly how to use and call them.

## How does the hugo converter work?

The algorithm follows the same steps Battaglia and al. 2013 use to calibrate its bias. First, it computes the density field and the redshift of reionization field from 21cmFAST (not applicable if the fields are imported). The density field is computed at the mean redshift of reionization, evaluated to be the mean of the computed redshift of reionization field. The fluctuations fields of both fields are taken. Fluctuation fields are computed with the average of their fields, showing the relative differences from the mean rather than the values themselves. Following, the power spectrum of both fields are computed, and the following bias feature is used to compute its value: 

![Screenshot from 2022-11-03 17-18-43](https://user-images.githubusercontent.com/59851566/200437285-aeebf956-d8b0-4bbd-878b-b4dec202b9fa.png)

After, The cross-spectra is computed, and the following formula is used to compute the cross-correlation:

![image](https://user-images.githubusercontent.com/59851566/206955949-a1c3418a-cb34-462f-8c36-271a60c13974.png)

The linear bias is then fitted using Markov Chains Monte Carlo (MCMC) to generate posterior distributions for the free parameters (alpha, b_0 and k_0). In this fitting, the cross-correlation is used to weight errors. Thirty-two walkers are launched across four dimensions (the basic error is the 4th fitted parameter: error = fitted_error/cross-correlations) for 5000 iterations. Walkers' behaviour, corner plots and best fits on the linear bias can be observed with the function. Since the resulting best-fitted parameters are posterior distributions (see Bayesian statistics), the mean of that distribution is used as the best-fit value. In contrast, the 16th and 84th percentiles are used for the confidence interval. 

The following schematic resumes the process followed by the algorithm in the calibration of the free parameters.

![SmartSelect_20221210-224839_Drive](https://user-images.githubusercontent.com/59851566/206955486-29e7a7c1-50df-479e-986f-f1784c612248.jpg)

For more information, internship reports from research classes further explain the process. Please note that the cross-correlation error weighting (a feature not yet implemented when writing the first report) is described in the second one. 

## Important to note: a word on units. 

In most of the literature, the spatial unit used for box length is Mpc / h, where h is the Hubble parameter. This makes the box h independent. 
However, 21cmFAST, just like the hugo converter, uses Mpc. 

The free parameter values (k_0) arre unit dependant!!! 

Therefore, if using a model with a unit of Mpc/h, the final parameter value must be corrected by a factor of h³ (Fourrier convention). This is the case for (I think) prof. James's Aguire TAU parameter algorithm. This is why the figures showing James's results have been corrected. 

## Uncertainties and further possible steps

The conversion of Mpc/h to Mpc in prof. Aguire's code is a noted uncertainty. Moreover, the elastic value of b_0 remains a not completely solved issue. Tests have been made, and it's clear that b_0 needs to be a fittable parameter for the model to work. Yielding slow variance to input parameters change, its value is, however, twice higher than the suggested fixed value in the Battaglia et al. paper. The paper says b_0 value was derived until it was equal to the inverse critical over-density threshold. However, the original derivation has not yet been found by prof. Liu: as the paper they cite doesn't explicitly derive it. Paul Laplante's z-reion has been changed to allow the variation of b_0. More details about this b_0 situation can be found in the research report.

Some functions exist to add James's z-reion results to the comparative study between hugo_converter's z-reion and 21cmFAST. Some of them are cleaned and added in the z_reion_comparison module, but some were commented out in the project_driver file. They can work, but time was missing to make a standardized code linking the codes. Moreover, some options (like modifying the MCMC number of walkers, their initial positions and the number of iterations) could be added. Moreover, although several tests have been run and results have been verified countless times, More testing functions could be added. Finally, I am pretty sure a more effective way of computing the redshift of reionization field exists and could be implemented in the hugo_converter to make the program more efficient. One hypothesis is trough lightcone functions.

## Docstrings and optional parameters of the two main functions

Every function contains detailed documentation explaining the main and imputable parameters. For simplicity purposes, here is the code's list of available options for the main function: `get_params_value`

```{python}
def get_params_values(box_len=143, box_dim=143, include_confidencerange=False, redshift_range=np.linspace(5, 18, 60),
                      nb_bins=20, density_field=None, zre_field=None, plot_best_fit=False, plot_corner=False,
                      return_zre_field=False, return_density=False, return_power_spectrum=False,
                      astro_params={"HII_EFF_FACTOR": 30.0}, flag_options={"USE_MASS_DEPENDENT_ZETA": False},
                      SIGMA_8=0.8, hlittle=0.7, OMm=0.27, OMb=0.045, POWER_INDEX=0.9665, find_bubble_algorithm=2):
    '''
    This function computes the linear bias free parameter values of z-reion
    :param box_len: [int] the spatial length of the desired box in Mpc (default is 143 Mpc which is equivalent to 100 Mpc/h)
    :param box_dim: [int] the dimension of the box (number of points per field) (default is 143 for a spatial voxel resolution of (1 Mpc/h)³
    :param include_confidencerange: [bool] return the confidence range (upper and lower limit) of the parameters. This corresponds to 68% of the       
                                           posterior distribution of each parameter
    :param redshift_range: [1D array] this is the redshift range used for the computation of the redshift of reionization. The more precise the range 
                                      (the more element in the array), the more precise/accurate the values of the parameter are, but the more 
                                      computational time it takes
    :param nb_bins: [int] the number of data points for the power spectrums and the bias (default 20). More can increase precision but reduce accuracy. 
                          Past work shows sweet point being the default 20
    :param density_field: [3D array] The density field used for the bias computation. None computes and uses 21cmFAST density field (default None)
    :param zre_field: [3D array] The redshift of reionization field used for the bias computation. None computes and uses 21cmFAST density field 
                                 (default None)
    :param plot_best_fit: [bool] Will plot the best fitted parameters over the computed bias if True (default True)
    :param plot_corner: [bool] Will plot the posterior distribution of the best fitted parameters if True (default True)
    :param return_zre_field: [bool] will return the redshift of reionization field if True (default True)
    :param return_density: [bool] will return the density field if True (default True)
    :param return_power_spectrum: [bool] will return the power spectrums of the density field and redshift of reionization field if True (default True)
    :param astro_params: [dict] a dictionary of all the wanted non-default astrophysical parameters on the form { input_param : value, ...} An 
                                extensive list of the usable astro parameters can be find here :      
                                https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :param flag_options: [dict] a dictionnary of all the wanted non-default flag options parameters on the form { flag_option : value, ...}. This 
                                include the use-mass_dependant_zeta function for the usage of astro parameters such as the turnover mass. An extensive                                   list of the usable flag options can be find here :           
                                https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :param SIGMA_8: [float] the cosmological value (default 0.8 )
    :param hlittle: [float] the cosmological value (default 0.7)
    :param OMm: [float] the cosmological value (default 0.27)
    :param OMb: [float] the cosmological value (default 0.045)
    :param POWER_INDEX: [float] the cosmological value (default 0.9665 )
    :param find_bubble_algorithm: [int] what method to use when finding the bubbles (default = 2)
    :return: [int or list] the values for the best-fitted free parameters alpha, b_0 and k_0, plus all other optional observable (the results gives a 
                           list if confidence interval are included)
    '''
```

And for the function parameter_2Dspace_run:

```{python}
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
    :param name_input1: [string] The name of the first changing astrophysical input (all the possible inputs can be found at : 
                                  https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html)
     Note! range1 must be in decreasing order for the object file to be like a normal cartesian plane
    :param range1: [list] the range of the desired first input values
    :param name_input2: [string] The name of the second changing astrophysical input (ex: HII_EFF_FACTOR for ionization efficiency (zeta))
    :param range2: [list] the range of the desired second input values
    :param file_name: [string] the name pf the file you want to save the file under (workds with directory to)
    :param redshift_range: [list] the redshift range at which to
    :param box_len: [int] the spatial length of the desired box in Mpc (default is 143 Mpc which is equivalent to 100 Mpc/h)
    :param box_dim: [int] the dimension of the box (number of points per field) (default is 143 for a spatial voxel resolution of (1 Mpc/h)³
    :param other_astro_params: [dict] if you want another astrophysical parameters to stay stable, but under a different value than the default one
    :param flag_options: [dict] a dictionnary of all the wanted non-default flag options parameters on the form { flag_option : value, ...}. This 
                                include the use-mass_dependant_zeta function for the usage of astro parameters such as the turnover mass. An extensive 
                                list of the usable flag options can be find here : 
                                https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html
    :param SIGMA_8: [float] the cosmological value (default 0.8 )
    :param hlittle: [float] the cosmological value (default 0.7)
    :param OMm: [float] the cosmological value (default 0.27)
    :param OMb: [float] the cosmological value (default 0.045)
    :param POWER_INDEX: [float] the cosmological value (default 0.9665 )
    :param find_bubble_algorithm: [int] what method to use when finding the bubbles (default = 2)
    :param include_zreion: [bool] will include z-reion computation and observables if True
    :param comp_brightness_temp: [bool] will compute the brightness temeperature and add it to the model if True
    :return: [2D array] an array containing objects for each of the varying variable run. Each object contains information about 21cmFAST and z-reion. 
                        The object structure and attribute can be found on the repo
    '''
```
## Final note

Please don't hesitate to reach out by email at: hugo.baraer@mail.mcgill.ca for any questions or inquiries. This work represents a year and a half of hard work and makes me proud. I hope you'll have a lot of fun using the hugo converter.
