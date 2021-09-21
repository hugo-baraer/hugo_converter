# EoR_research
This repository is used for Hugo Baraër's research with the cosmic dawn group at McGill, supervised by prof. Adrian Liu. Currently, it includes the PHYS 449 research project, a parameters comparison in two different models: 21cmFAST and Battaglia and al. (2013) 

## Motivation

Reionization is the period of the universe following the dark ages, where gas reionized, leading to the birth of the first start. This period plays a crucial role in our understanding of the structure of our universe. Before a model proposed by Battaglia et al. in 2013, direct simulations of the evolution of the ionization field in large volumes ( approximately bigger than (Gpch^-1)^3) were not attainable. Using previous precise short-scale models (Trac and Pen 2004) (Trac and Chen 2007), they quantify the correlation between reionization redshift and density using two points statistics in Fourier space. This way, an N-body simulation can be converted into statistical parameters. Therefore, the model can compute the evolution of the 3D ionization field in large volumes with a low computational time. On the downside, the parameters computed in this model are purely statistical, with initial values that are not well defined. 

Another model called 21cm Fast is in the middle between fully hydrodynamical and semi-analytical models like Battaglia et al. This model incorporates physical parameters but has more considerable computational speed.

## Project description

This project aims to link the statistical parameters of Battaglia and al. to the model of 21cm fast. This way, initial values of those parameters can be established, facilitating the model's usage. 

The model proposed by Battaglia et al. starts from a P³M N-body code to generate the over-density field. The Fourier transform is then taken for a particular redshift. Next, the statistical parameter acting as an N-body simulation with radiative transfer and hydrodynamic is multiplied to that over-density field in momentum space, giving the over-redshift in momentum space. An inverse Fourier transform is then performed, and the over redshift is shifted back to the redshift field. The following diagram presents the Battaglia model and its steps towards plotting the reionization redshift field.

![241468409_457949906006586_5301099429280127917_n](https://user-images.githubusercontent.com/59851566/134114580-0f89fb22-307d-4a1e-a9b4-d22853eb8747.jpg)

21cmFast has an option to generate density field and cosmic redshift field through different simulation processes. By generating both fields and reverse engineering the process shown in the figure below , it is possible to get a value for the parameters from Battaglia et al. 


## directory content

### FTT_overdensity

This module takes the Fourrier transform of the over-density. (details to come)

### FTT_overredshift

This module takes the Fourrier transform of the over-redsfhit. (details to come)

### compute_W

This module computes the parameter acting between the two transorm (What acts as  W factor in Battaglia). (details to come)

### find_params

This module computes the MCMC algorithm necessary to generate posterior distributions of the wanted parameters. (details to come)


### standard driver 

This file is the driver to run to execute the different modules. In this driver file, all the different components of the project are launched. 

