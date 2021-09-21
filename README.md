# EoR_research
This repository is used for Hugo Baraër's research with the cosmic dawn group at McGill, supervised by prof. Adrian Liu. Currently it includes the PHYS 449 research project, a parameters comparison in two different models : 21cmFAST and Battaglia and al. (2013) 

## Motivation

Reionization is the period of the universe following the dark ages, where gas reionized, leading to the birst of the first start. This period plays a crucial role in our understanding of the structure of our universe. Before a model proposed by Battaglia et al. in 2013, direct simulations of the evolution of the ionization field in large volumes ( approximatly bigger than (Gpch^-1)^3) was not attainable. Using previous precise on short short-scale models (Trac and Pen 2004) (Trac and Chen 2007), they quantify the correlation between reionization redshift and density using two points statistics in Fourier space. This way, a N body simulation can be converted into statistical paramters. Therefore, the model can compute the evolution of the 3D ionization field in large volumes, with a low computational time. On the downside, the parameters computed in this model are purely statistical, with initial values that are not well defined. 

Another model called 21cm Fast is in the middle between fully hydrodynamical models and semi-analytical models like Battaglia et al. This models incorporate physical parameters, but has bigger computational speed.

## Project description

The goal of this project is to link the statistical parameters of Battaglia and al. to the model of 21cm fast. This way, initial values of those parameters can be establish, hence faciliting the usage of the model. 

the model proposed by Battaglia et al. starts from a P³M N-body code to generate the over-density field. The fourrier transform is then taken for a particular redshift, and the statistical parameter acting as a N-body simulation with radiative transfer and hydrodynamic is multiplied to that over-density field in momentum space, giving the over-redshift in momentum space. An inverse Fourier transform is then performed, and the over redshift is shifted back to the redshift field. The following diagram presents the Battaglia model and its steps towards plotting reionization redshift field.

![241468409_457949906006586_5301099429280127917_n](https://user-images.githubusercontent.com/59851566/134114580-0f89fb22-307d-4a1e-a9b4-d22853eb8747.jpg)

Since 21cmFast have an option to generate density field, as well as cosmic redshift field, through different simulation processes. By creating 


## directory content

###

This module takes out 

### standard driver 

This is the driver to run to exectute the different modules. This is where all the different components of the project are launched. 

