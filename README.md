# Baraer's converter

#### Generate z-reion's linear bias free parameters value based on 21cmFAST physical parameters inputs.

## Author
This repository and its content is credited to Hugo Baraer [@hugo-baraer](https://github.com/hugo-baraer) and his research with the cosmic dawn group at McGill University, supervised by prof. Adrian Liu [@acliu](https://github.com/acliu). 

It is the results of 3 undergraduate physics research classes, and a SURA summer internship. 
Currently, it includes the PHYS 449 research project, a parameters comparison in two different models: 21cmFAST and Battaglia and al. (2013) 

## explanation

z-reion provides a statistical linear bias linking density fields to redshift of reionization fields. The above equation represent the role of the linear bias (b_mz) in linking the mean centered version of the two fields in momentum space.

![Screenshot from 2022-06-16 22-48-35](https://user-images.githubusercontent.com/59851566/200427369-d2d822ad-3a91-4672-b56f-593a929a1064.png)

The linear bias in it's simpliest form is expressable as: 

![Screenshot from 2022-06-16 22-50-42](https://user-images.githubusercontent.com/59851566/200427950-3221477b-d322-41c2-a602-e76f3a96064c.png)

Using z-reion allows for a computationnaly quick generation of a redshfit of reionization field. However, since z-reion is a semi-analytical model relying on a statistical term, it has no physical parameters' inputs (astrophyscial, cosmological, etc.). [21cmFAST](https://github.com/21cmfast/21cmFAST) provides a model with inputable physical parameters. Plugging in 21cmFAST inputs, Hugo's converter can be used to fit for the values of z-reion's bias parameters. Plugged in Paul Laplante [@paplant](https://github.com/plaplant) z-reion python implementation (the model was originally designed and proposed by:  Battaglia and al. (2013)), redshfit of reionization fields can then be quickly generated from density fields. 


## Features

## Installation

Reionization is the period of the universe following the dark ages, where the hydrogen composing the intergalactic medium reionized, following the birth of the first start. This period plays a crucial role in our understanding of the structure of our universe. Before a model proposed by Battaglia et al. in 2013, direct simulations of the evolution of the ionization field in large volumes ( approximately bigger than (Gpch^-1)^3) were not attainable. Using previous precise short-scale models (Trac and Pen 2004) (Trac and Chen 2007), they quantify the correlation between reionization redshift and density using two points statistics in Fourier space. This way, an N-body simulation can be converted into statistical parameters. Therefore, the model can compute the evolution of the 3D ionization field in large volumes with a low computational time. On the downside, the parameters computed in this model are purely statistical, with initial values that are not well defined. 

Another model called 21cm Fast is in the middle between fully hydrodynamical and semi-analytical models like Battaglia et al. This model incorporates physical parameters but has more considerable computational speed.

## How those it work?

This project aims to link the statistical parameters of Battaglia and al. to the model of 21cm fast. This way, initial values of those parameters can be established, facilitating the model's usage. 

The model proposed by Battaglia et al. starts from a P³M N-body code to generate the over-density field. The Fourier transform is then taken for a particular redshift. Next, the statistical parameter acting as an N-body simulation with radiative transfer and hydrodynamic is multiplied to that over-density field in momentum space, giving the over-redshift in momentum space. An inverse Fourier transform is then performed, and the over redshift is shifted back to the redshift field. The following diagram presents the Battaglia model and its steps towards plotting the reionization redshift field.

21cmFast has an option to generate density field and cosmic redshift field through different simulation processes. By generating both fields and reverse engineering the process shown in the figure below , it is possible to get a value for the parameters from Battaglia et al. 


![battaglia_process](https://user-images.githubusercontent.com/59851566/200426827-45335b46-d89c-4a1c-a462-fca73e590b66.jpg)

## directory content

The algorithm has functions seperated in different modules. Having more than 85 different functions, here is a brief decription of the repartition.  

### FTT

This module takes the Fourrier transforms of both fields, and contains the necesseray functions for plotting and computing.

### Gaussian_testing

This module is made for testing the fft process with 3D gaussian testing

### z_re_field.py

This module computes the redshift of reionization field (as well as the overredshift of rieonization)with 21cmFAST data

### statistical_analysis.py

This module computes the MCMC algorithm necessary to generate posterior distributions of the wanted parameters, as well as the shell average to transform the 3D field into 1D field

### z_reion_comp.py

### plot_params.py

### project_driver 

This file is the driver to run to execute the different modules. In this driver file, all the different components of the project are launched. 

