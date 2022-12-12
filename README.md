# Hugo Converter

#### Find z-reion's linear bias-free parameters value based on 21cmFAST physical parameters inputs or any density and redshift of reionization fields.

## Author
This repository and its content are credited to Hugo Baraer [@hugo-baraer](https://github.com/hugo-baraer) and his research with the cosmic dawn group at McGill University, supervised by prof. Adrian Liu [@acliu](https://github.com/acliu). 

It results from one and a half years of work : 3 undergraduate physics research classes and a SURA summer internship. 

## Features

* Computes the best-fit value of the three free parameters of the linear bias, as well as their confidence interval (68% of the posterior distribution)
* Works with any 21cmFAST inputs (different inputs available [here](https://21cmfast.readthedocs.io/en/latest/_modules/py21cmfast/inputs.html) )
* Option to use your own density and/or redshift of reionization fields to compute the parameters' value.
* Contains all the necessary analysis functions, including plotting and the computation of several observables (power spectrums, ionization histories, brightness temperatures, TAU parameters, etc.)
* Possibility to run parameter space studies (vary inputs simultaneously and see its effect on parameters or observables) in 1D and 2D
* Easily generate and save 21cmFAST fields (density, ionization maps, and brightness temperature)

## brief explanation

The `z-reion` model provides a statistical linear bias linking density fields to a redshift of reionization fields. The above equation represents the role of the linear bias (b_mz) in linking the fluctuations of the two fields in momentum space.

![Screenshot from 2022-06-16 22-48-35](https://user-images.githubusercontent.com/59851566/200427369-d2d822ad-3a91-4672-b56f-593a929a1064.png)

The linear bias in its simplest form is expressable as: 

![Screenshot from 2022-06-16 22-50-42](https://user-images.githubusercontent.com/59851566/200427950-3221477b-d322-41c2-a602-e76f3a96064c.png)

Using z-reion allows for a computationally quick generation of a redshift of reionization field. However, since z-reion is a semi-analytical model relying on a statistical term, it has no physical parameters' inputs (astrophysical, cosmological, etc.). [21cmFAST](https://github.com/21cmfast/21cmFAST) provides a model with imputable physical parameters. Plugging in `21cmFAST` inputs, `Hugo's converter` can be used to fit the values of z-reion's bias parameters. Plugged in Paul Laplante [z-reion](https://github.com/plaplant/zreion) python implementation (the model was initially designed and proposed by:  Battaglia and al. (2013)), redshift of reionization fields can then be quickly generated from density fields. 

A more detailed explanation is provided in the folder `docs`


## Installation

Installing the package can be done by cloning the repo, 

'git clone https://github.com/hugo-baraer/hugo_converter.git'

Going into the repo and then running:

`pip install . `

This will initialize the 'setup.py' file and install the package under the name 'hugo_converter'
Dependencies will be handled automatically by pip if not already installed. Here is a list of packages installed by the setup

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

To install `z-reion`, follow a similar process. from the EoR_research change directory to the z-reion subfolder located in the hugo_converter folder.

`cd ./hugo_converter/zreion/`

Then, simply run again: 

`pip install . `

This will install the z-reion package. The hugo-converter installation covers its dependencies.

You are now all set! to test you can test the most basic version of the Hugo converter: 

`import hugo_converter as hc`

`hc.get_params_value()`


## directory content
In the main directory, installation files and a gitignore file are present. In addition, there is four subfolders. Another repo, called `hugo_converter_figures`, contains all the important figures and a jupyter notebook to generate them again from data. 

### docs

This directory contains the necessary documentation to understand how the hugo converter works, along with numerous examples of how to use it, with a jupyter notebook demonstration. 

### data

This directory contains all the 2D variation studies made, along with some cached fields, such as an ultraprecise redshift of reionization field from 21cmFAST and some other cached fields that could be useful. 

### hugo_converter

The algorithm has functions separated into different modules. Having more than 85 different functions, here is a brief description of the repartition.  

#### z_re_field.py

This module computes the redshift of reionization field (as well as the over redshift of reionization)with 21cmFAST data.

#### statistical_analysis.py

This module computes the MCMC algorithm necessary to generate posterior distributions of the wanted parameters, as well as the shell average to transform the 3D field into 1D field

#### z_reion_comp.py

This module contains all the necessary analysis functions for comparison of observables (like ionization history of brightness temperature)

#### plot_params.py

This module contains all the functions to plot observables for single runs.

#### project_driver 

In this driver file, all the different components of the project are launched. This file contains two main functions: 

get_params_values()

parameter_2Dspace_run

### testing modules 

Two modules to test power spectrum computation are also included:
#### FTT.py

This module takes the Fourier transforms of both fields and contains the necessary functions for plotting and computing.

#### Gaussian_testing.py

This module was done for testing the fft process with 3D gaussian testing.

