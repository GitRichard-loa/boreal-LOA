## 27 June 2023
### In ./forward_module/spheroid/spheroid.py
1. Add ***get_optdust()*** to consider the spectral dependence of dust (based on the regression analysis of the measurements by Di Biagio et al. (2019).)

### In ./BOREAL.py
1. In ***inversion()***, replace parameter ***type_abs*** with ***aero_type***, which has to be one of the following 
strs: 'bba', 'dust', 'urban', 'ss' (for see salt). Different aero_types have different a priori constraints on CRI. See ***get_default_config()***
2. set ***epsilon1*** to 1/3 of the input measurement error based on the Gaussian error assumption.
The default values of ***epsilon1*** are: 0.03 for ext and bac (355, 532); 0.05 for depol; 0.06 for bac (1064).
3. It is found that using the new coefficient for deriving ***k532*** in ***ih.py*** sometimes makes ***n*** exceed the LUT boundary (overflow).
This problem can be solved by expanding ***epsilon1***.
Therefore, everytime overflow happens, enlarge ***epsilon1*** by a factor of 1, 2, 3, 4 (not larger than 4) and try again.
4. In ***retrieval*** returned by ***inversion()***, add 'VSD_IS' which is a collection of every individual solution.

## 28 June 2023
### In ./forward_module/spheroid/spheroid.py
1. When one wants to call the sphere model, it is quite likely to misspell the model name ***sphere*** as ***shpere***.
In previous code, this will lead to call the spheroid model. To avoid such mistake, in the updated code, if the input model
name is neither 'sphere' nor 'spheroid', a ***FwdModelException*** will be raised.

## 3 October 2023
### In ./BOREAL.py
1. In ***inversion()***, the structure of the code has been improved to make it more compact.
2. In ***__do_iter()***, change the strategy of iteration, see Thesis for more detail.

## 27 October 2023
### add and complete codes for automated test (test.py)
### spheroid.py:
1. improve the function ***get_opt***: ***depol_wave*** does not have to be a subset of ***bac_wave***.
If this happens in previous versions, a **KeyError** exception will raise.
2. like ***get_opt***, improve the function ***get_optdust*** and change the name to ***get_opt_spectralmI***. Relative
changes are also made in **BOREAL.py**.
### BOREAL.py:
1. improve the codes: (1) print the inverted measurements; (2) always plot 4 panels in ***plot_fit*** (recalculated and
measured (if available) ext, bac, lr, depol); (3) in the generation of the initial VSD, if ext355 is not available, fit 
ext532 instead, if ext532 is still not available, fit ext1064 instead; (4) the input depol_meas can be absent, empty dict,
or non-empty dict, while the input ext_meas and bac_meas can be non-empty dict with arbitrary length; (5) abandon the
argument "*lr_err*" in ***inversion***, it is calculated from *ext*, *bac*, *ext_err* and *bac_err* according to the 
relationship *lr=ext/bac*, instead.
2. ***derivative***: change the absolute step for calculating the Jacobian from *1e-5* to *1e-7*. It is found that the
previous value seems too large: the difference of retrieve['VSD_mean'] when ***get_optdust*** and ***get_opt_spectralmI***
are used is up to 1e-6 (which should be less than 1e-7). After use *1e-7*, the difference is less than 1e-7.

## 24 January 2024
1. improve the appearance and fix typing errors of the online site page.
2. paste a _readme_ file on the site.
3. prescribe the total number of input measurements has to be 3 at least, otherwise, a BOREAL exception will raise.
4. change the aerosol type name "biomass burning" to "absorbing", "urban" and "sea salt" to "non-absorbing".

## 22 October 2025 - Big update to v0.5.0
### Integrate IH model into BOREAL
1. In **forward_module**, add **tamuDUST** and ***ih.py***
2. The IH model can be called by *BOREAL.inversion()* by setting *model='ih'*
3. Dust mI spectral dependency is taken into account.
### Add BOREAL-PC method
1. PC for "Parameterized Constrained".
2. ***BOREAL_PC.py*** is added. Related instructions are appended in ***README***
### Code ameliorating
1. In ***BOREAL.py***: (1) change the name *FwdModelException* to *BOREAL_FwdModelException*; (2) refine the code.
2. In ***spheroid.py***, combine *get_opt()* and *get_opt_spectralmI()* to one *get_opt()*
3. In ***runBOREAL.py***: (1) refine the code; (2) add two dust cases for testing (BOREAL & BOREAL-PC), one simulated case for the BOREAL-PC testing.
4. The reference outputs from the original testing are replaced with the ones from the modified testings, and collected in a fold named **reference** under **run_example**.
5. ***README.md*** file is updated.
