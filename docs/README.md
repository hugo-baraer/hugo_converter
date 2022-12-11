# Hugo converter documentation

This folder contains the necessary documentation to understand everything about the fmmous and incredible hugo converter. In the subfolder presentation_slides, you'll find the slides for the 3 presentation given at the Comsic dawn group. In the the final_report folder, you'll find the final report for two research classes done during the first an second semester of work. Finally, the masterpiece,  you'll fin presentation.ipynb, a jupyter notebook showing the main functions, their several options and exactly how to use and call them.

## How does the hugo converter works?

The algorithm follows the same steps Battaglia and al. 2013 uses to calibrate it's bias. First, it computes both the density field and the redshfit of reionization field from 21cmFAST (not applicable if the fields are imported). The fluctuations fields of both fields is taken. Fluctuation fields
are computed with the average of their fields, showing the relative differences from the mean rather than the values themselves. The power spectrum of both fields are computed, and the following bias feature is used to computed it's values: 

![Screenshot from 2022-11-03 17-18-43](https://user-images.githubusercontent.com/59851566/200437285-aeebf956-d8b0-4bbd-878b-b4dec202b9fa.png)

The linear bias is then fitted using Markov Chains Monte Carlo, to generate posterior distributions for the free parameters (alpha, b_0 and k_0). In this fitting, the cross correlation is used to weight errors. The following schematic resumes the process followed by the algorithm in the calibration of the free parameters.

![battaglia_process](https://user-images.githubusercontent.com/59851566/200426827-45335b46-d89c-4a1c-a462-fca73e590b66.jpg)

For more information, internship reports from research classes present here further explaines the process. Please note that the cross-corelation error weighting (a tecninc not yet implemented in the first report), is described in the second one.

## Uncertainties


## Important to note: a word on units. 

h versus h_bar. The parameters will have units of which you choose
