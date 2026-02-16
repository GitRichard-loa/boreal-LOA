# BOREAL package for aerosol microphysical property retrieval from lidar measurements
## General description 
The **BOREAL** (**B**asic alg**O**rithm for **RE**trieval of **A**erosol with **L**idar) algorithm is developed by 
the Laboratoire d'Optique Atmosphérique, a joint research unit of the University of Lille and CNRS.
This package retrieves particle volume size distribution (VSD) and complex refractive index (CRI = mR -imI) from 
lidar-derived extinction + backscattering (or + depolarisation) properties. Total volume concentration (Vt), 
effective radius (reff) and single-scattering albedo (SSA) are then calculated from the retrieved VSD and CRI.

## Data policy
If you utilize the BOREAL retrieval products for publication purposes, we kindly request you to cite the paper listed 
in **Citation** and acknowledge the contribution of "University of Lille/CNRS/Laboratoire d'Optique Atmosphérique".
Additionally, we encourage you to consider offering co-authorship to the scientists who contributed to the development 
of BOREAL, if their involvement is relevant to your work. Your recognition and collaboration contribute to the 
advancement of scientific research and the acknowledgment of the efforts invested in the development of these 
resources for the community.

## Citation
If you use this software in your work, please cite the software as  
> [to be complete]

and the following papers  
> * Chang, Y., Hu, Q., Goloub, P., Veselovskii, I., and Podvin, T.: Retrieval of Aerosol Microphysical Properties from Multi-Wavelength Mie–Raman Lidar Using Maximum Likelihood Estimation: Algorithm, Performance, and Application, Remote Sens., 14, 6208, https://doi.org/10.3390/rs14246208, 2022.  
> * Chang, Y., Hu, Q., Goloub, P., Podvin, T., Veselovskii, I., Ducos, F., Dubois, G., Saito, M., Lopatin, A., Dubovik, O., and Chen, C.: Retrieval of microphysical properties of dust aerosols from extinction, backscattering and depolarization lidar measurements using various particle scattering models, Atmos. Chem. Phys., 25, 6787–6821, https://doi.org/10.5194/acp-25-6787-2025, 2025.

## Acknowledgments  
The scattering properties of irregular particles are obtained from the TAMUdust2020 database (<https://zenodo.org/record/4711247>) with reference: Saito, M., P. Yang, J. Ding, and X. Liu (2021), A comprehensive database of the optical properties of irregular aerosol particles for radiative transfer simulations, J. Atmos. Sci., 78, 2089–2111. The scattering properties of spherical and spheroidal particles are obtained from the GRASP Spheroid-package (<https://www.grasp-open.com/products/spheroid-package-release/>) with the reference: Dubovik, O., A. Sinyuk, T.Lapyonok, B. Holben, M. Mishchenko, P. Yang, T. Eck, H. Volten, O. Munoz, B. Veihelmann, W. van der Zande, J.-F. Leon, M. Sorokin, I. Slutsker (2006), Application of spheroid models to account for aerosol particle nonsphericity in remote sensing of desert dust, J. Geophys. Res., 111, D11208, doi:10.1029/2005JD006619.

## License
This project is licensed under the BSD-3-Clause with extra terms (see **LICENSE.txt** attached with the project).

## Structure of the package
Scripts and datasets are contained in **./boreal**, where the folder **forward_module** includes the implementations of the sphere, spheroid and ih models. **BOREAL.py** and **BOREAL_PC.py** call the forward models
and realize the inverse process.

## Installation
### Set up a Python 3.9+ environment
```bash
# in shell, go to the objective directory then type
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip 
```

### There are two ways to install the package.  
1. You can download the .tar.gz file from <https://pypi.org/project/boreal-LOA/#files>, or <https://github.com/GitRichard-loa/boreal-LOA.git>, unarchive the file and copy all the contents in the root directory to the objective directory where the python environment has been set, then in the objective directory, type
```bash
python3 -m pip install -r requirements.txt
```
2. Alternatively, you can install the package from PyPI. In the objective directory, type
```bash
python3 -m pip install boreal-LOA
```

The package can be imported by python:
```python
from boreal import BOREAL, BOREAL-PC
```

## Use of BOREAL.py
The python script implementing the BOREAL method. To run the retrieval, in a python script or an interactive shell, type the following commands:  
```python
from boreal import BOREAL

# perform the retrieval
retrieval, fit = BOREAL.inversion(**args)  # use help function to check more instructions

# visualise the results
BOREAL.plot_fit(**args)  # plot the fitting
BOREAL.plot_rtv(**args)  # plot the retrieval
BOREAL.export_txt(**args)  # output the txt file
```

Mandatory arguments in `BOREAL.inversion()`:
* *ext*: dict, spectral extinction coefficient, the keys (str) are wavelength in nm, the values are corresponding measurements (float) in 1/Mm
* *bac*: dict, spectral bac. coef., the keys (str) are wavelength in nm, the values are corresponding measurements (float) in (Mm*sr)^(-1)
* *aero_type*: str, 'dust', 'bba', 'urban' or 'ss' (sea salt), a priori knowledge of aerosol type
* *model*: str, 'sphere', 'spheroid' or 'ih', forward model (scattering model) used in the inversion

Optional arguments:
* *depol*: None or dict (default=None), particle spectral depolarization ratio, the keys (str) are wavelength in nm, the values are corresponding measurements (float) (unit of 1)
* *ext_err*: None or dict (default=None), maximum measurement error in ext (three times of measurement std). None for default values.
* *bac_err*: None or dict (default=None), maximum measurement error in bac (three times of measurement std). None for default values.
* *depol_err*: None or dict (default=None), maximum measurement error in depol (three times of measurement std). None for default values.
* *config*: None or dict (default=None), customized configuration for implementing the retrieval

## Use of BOREAL_PC.py
The python script implementing the BOREAL-PC method which retrieves parameterized VSD and CRI with the aid of a priori constraints from historical in situ measurements. To run the retrieval, in a python script or an interactive shell, type the following commands: 
```python
from boreal import BOREAL_PC

# organise the input optical data
opt_harmonized = BOREAL_PC.harmonize_opt_format(ext, bac, depol)   # ext, bac, depol are dictionaries with keys=wavelength and values=values. e.g., ext={'355': value_355, '532': value_532}

# perform a retrieval
boreal_pc_instance = BOREAL_PC.Retrieval_bimodal(opt_harmonized)
boreal_pc_rtv = boreal_pc_instance.do_retrieval(**args)

# output results in txt format
BOREAL_PC.export_txt(**args)
``` 

Input arguments for `BOREAL_PC.harmonize_opt_format(**args)`:
* *ext*: dict, same as that input to `BOREAL.inversion(**args)`, but the wavelengths have to be '355' and '532'
* *bac*: dict, same as that input to `BOREAL.inversion(**args)`, but the wavelengths have to be '355', '532' and '1064'
* *depol*: dict,same as that input to `BOREAL.inversion(**args)`, but is mandatory and the wavelengths have to be '355', '532' and '1064'  

Note: 
1. Since BOREAL-PC is specially designed for dust retrieval, the argument *aero_type* makes no sense
2. Only the IH model is available.
3. To ensure acceptable retrieval accuracy, the complete optical dataset (i.e., 2a+3b+3d) is required. The accuracy of inverting deficient dataset needs further evaluation.