# R code for the two epidemiological data analysis presented in my thesis

This repository contains the code used in Janet's master thesis. Each file contains the model code, an aritificial data generating process, and the functions to run the code. The two files ech corresponds to one data analyis in the thesis.

We use the model coded in the "AS data analysis code" file to explore the possible inequality existed in treatment decisions among patients in Quebec who diagnosed with aortic stenosis and needed valve replacement. In addition to the variable selection processed through the regularized Horseshoe prior, we added a binary adjacency structure to model the differences between CLSC regions.

Finally, the "Missing data imputation with HIV data" file contains the code we used to construct a Bayesian hierarchical model that simultaneously impute missing data estimates the correlations between personal background, behaviours and HIV pravelence. The missing values are imputed in a fully Bayesian way.
