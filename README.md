# Hugo Converter

#### Find z-reion's linear bias free parameters value based on 21cmFAST physical parameters inputs, or from any density and redshfit of reionization fields.

## Author
This repository and its content is credited to Hugo Baraer [@hugo-baraer](https://github.com/hugo-baraer) and his research with the cosmic dawn group at McGill University, supervised by prof. Adrian Liu [@acliu](https://github.com/acliu). 

It is the results of one an a half year of work : 3 undergraduate physics research classes, and a SURA summer internship. 

## Features

* Computes the best-fit value of the three free parameters of the linear bias, as well as their confidence interval (68% of the prosterior distribution)
* Works with any 21cmFAST inputs (different inputs available [here](https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html) )
* Option to use your own density and/or redshfit of reionization fields to compute the parameters values.
* Contains all the necessary analysis functions including plotting and the computation of several obsevables (power spectrums, ionization histories, brightness temperatures, TAU parameters, etc.)
* Possibility to run parameter space studies (vary simultaneaously inputs and see it's effect on parameters or observables) in 1D and 2D
* Easily generate and save 21cmFAST fields (density, ionization maps, and brightness temperature)

## brief explanation

The `z-reion` model provides a statistical linear bias linking density fields to redshift of reionization fields. The above equation represent the role of the linear bias (b_mz) in linking the fluctuations of the two fields in momentum space.

![Screenshot from 2022-06-16 22-48-35](https://user-images.githubusercontent.com/59851566/200427369-d2d822ad-3a91-4672-b56f-593a929a1064.png)

The linear bias in it's simpliest form is expressable as: 

![Screenshot from 2022-06-16 22-50-42](https://user-images.githubusercontent.com/59851566/200427950-3221477b-d322-41c2-a602-e76f3a96064c.png)

Using z-reion allows for a computationnaly quick generation of a redshfit of reionization field. However, since z-reion is a semi-analytical model relying on a statistical term, it has no physical parameters' inputs (astrophyscial, cosmological, etc.). [21cmFAST](https://github.com/21cmfast/21cmFAST) provides a model with inputable physical parameters. Plugging in `21cmFAST` inputs, `Hugo's converter` can be used to fit for the values of z-reion's bias parameters. Plugged in Paul Laplante [z-reion](https://github.com/plaplant/zreion) python implementation (the model was originally designed and proposed by:  Battaglia and al. (2013)), redshfit of reionization fields can then be quickly generated from density fields. 

A more detailled explanation is provided in the folder `docs`


## Installation

Installing the package can be done by cloning the repo, 

'git clone https://github.com/hugo-baraer/hugo_converter.git'

Going into the repo, and then running:

`pip install . `

This will initialize the 'setup.py' file and intall the package under the name 'hugo_converter'
Dependencies will be handled automatically by pip if not already installed. Here is a list of packages installed by the set up

`astropy`
`corner`
`emcee`
`imageio`
`matplotlib`
`numpy`
`pyfftw`
`powerbox`
`scipy`
`tqdm`

This full list of dependencies can be found in the setup.cfg file

#### Please Note!: 

The packages [`21cmFAST'](https://github.com/21cmfast/21cmFAST) and [z-reion](https://github.com/plaplant/zreion) are not included in this instalation. However, z-rieon can be installed from this package

### z-reion sub-installation

To install `z-reion`, follow a similar process. from the EoR_research change directory to the z-reion subfolder loacted in the hugo_converter folder

`cd ./hugo_converter/zreion/`

Then, simply run again: 

`pip install . `

This will install the z-reion package. Its dependancies are covered by the hugo-converter installation

You are now all set! to test you can test the most basic version of the Hugo converter: 

`import hugo_converter as hc`

`hc.get_params_value()`

## How those it work?

The algorithm follows the same steps Battaglia and al. 2013 uses to calibrate it's bias. First, it computes both the density field and the redshfit of reionization field from 21cmFAST (not applicable if the fields are imported). The fluctuations fields of both fields is taken. Fluctuation fields
are computed with the average of their fields, showing the relative differences from the mean rather than the values themselves. The power spectrum of both fields are computed, and the following bias feature is used to computed it's values: 

![Screenshot from 2022-11-03 17-18-43](https://user-images.githubusercontent.com/59851566/200437285-aeebf956-d8b0-4bbd-878b-b4dec202b9fa.png)

The linear bias is then fitted using Markov Chains Monte Carlo, to generate posterior distributions for the free parameters (alpha, b_0 and k_0). In this fitting, the cross correlation is used to weight errors. The following schematic resumes the process followed by the algorithm in the calibration of the free parameters.

![battaglia_process](https://user-images.githubusercontent.com/59851566/200426827-45335b46-d89c-4a1c-a462-fca73e590b66.jpg)

For more information, internship reports from research classes present here further explaines the process. Please note that the cross-corelation error weighting (a tecninc not yet implemented in the first report), is described in the second one.

## Important to note: a word on units. 

h versus h_bar. The parameters will have units of which you choose

## directory content
In the main directory, installation files and gitignore file are present. In addition, there is four subfolder. Another repo exists called `hugo_converter_figures` contains all the important figures along with a jupyter notebook to generate them again from data. 

### docs

This directory contains ...

### data

This al

### hugo_converter

The algorithm has functions seperated in different modules. Having more than 85 different functions, here is a brief decription of the repartition.  

#### z_re_field.py

This module computes the redshift of reionization field (as well as the overredshift of rieonization)with 21cmFAST data

#### statistical_analysis.py

This module computes the MCMC algorithm necessary to generate posterior distributions of the wanted parameters, as well as the shell average to transform the 3D field into 1D field

#### z_reion_comp.py

This module contains all the necessary analysis functions for comparison of observable (like ionazation history of brightness temperature)

#### plot_params.py

This module contains all the functions to plot observables for single runs

#### project_driver 

In this driver file, all the different components of the project are launched. This file contains the two main functions: 

get_params_values()

parameter_2Dspace_run

### testing modules 

two modules to test power spectrum computation are also included :
#### FTT.py

This module takes the Fourrier transforms of both fields, and contains the necesseray functions for plotting and computing.

#### Gaussian_testing.py

This module is made for testing the fft process with 3D gaussian testing

