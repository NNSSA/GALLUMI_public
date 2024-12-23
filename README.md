<p align="center">
  <img src="GALLUMI_logo.png" alt="GALLUMI Logo" width="50%" />
</p>

# GALLUMI: A Galaxy Luminosity Function Pipeline for Cosmology and Astrophysics

Authors: Nashwan Sabti, Julian Muñoz, and Diego Blas

GALLUMI is a likelihood code that allows to extract cosmological and astrophysical parameters from the UV galaxy luminosity function. The code is implemented in the MCMC sampler [MontePython] and can be readily run in conjunction with other likelihood codes. Details of the code are provided in [2110.13161] and [2110.13168].

This GitHub page contains 1) the GALLUMI likelihood code, 2) scripts to generate mock data, and 3) script to create plots.

## Basic Usage of GALLUMI

- Download the [MontePython] MCMC sampler. 
- Move the contents of MontePython_files to the MontePython folder. 
    - MontePython_files/Bestfits, MontePython_files/Covmats and MontePython_files/Param_files go in the main MontePython folder
    - Create a new folder called "UVLF" in MontePython/data/ and move the contents of MontePython_files/Data/ in there
    - Move the contents of MontePython_files/Likelihoods/ to MontePython/montepython/likelihoods/
- Run the likelihood code with e.g. 
    ```sh 
    python2 montepython/MontePython.py run --conf default.conf --param Param_files/UVLF_HST_ST_model1.param --bestfit Bestfits/UVLF_HST_ST_model1.bestfit --covmat Covmats/UVLF_HST_ST_model1.covmat -f 0.5 --output UVLF_HST_ST_model1
    ```
- In some cases, classy.pyx in CLASS should be modified. For example, when varying the power-spectrum amplitude, it'll be necessary to include the following lines in the function "get_current_derived_parameters":
  ```
    elif name == 'pk1p06':
        value = self.pk_lin(1.06, 0.)
    elif name == 'pk4p7':
        value = self.pk_lin(4.7, 0.)
  ```
## Example Plot
All plots in the papers [2110.13161] and [2110.13168] can be reproduced using the scripts provided in the Scripts/Plotting/ folder.

![sampleplot](sampleplot.png)

## Using GALLUMI in Publications
If you make use of the GALLUMI code in your publication, please cite the papers [2110.13161] and [2110.13168].
Chains are available upon request. 

## Contact
Please email nash.sabti@gmail.com for any questions :)


[MontePython]: <https://github.com/brinckmann/montepython_public>
[2110.13161]: https://arxiv.org/abs/2110.13161
[2110.13168]: https://arxiv.org/abs/2110.13168
