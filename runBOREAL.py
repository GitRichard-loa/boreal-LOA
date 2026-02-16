#!.venv/bin/python3

# an example python script about running BOREAL
from boreal import BOREAL, BOREAL_PC
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'run_example')

# Reference cases
# a simulated case Vt=1, Rc=2, Sc=1.7, fvf=0.01, mR=1.49, mI355=0.003 (for BOREAL-PC testing)
'''case_name = 'SimulatedCase_for_BOREALPC'
ext_sim = {'355': 1.4512879, '532': 1.44293407}
bac_sim = {'355': 0.02577472, '532': 0.03138802, '1064': 0.02802447}
depol_sim = {'355': 0.43155294, '532': 0.4418685, '1064': 0.37096174}'''

# a simulated case for BOREAL, model='spheroid', aero_type='dust'
'''case_name = 'SimulatedCase_for_BOREAL'
ext_sim = {'355': 58.0121, '532': 61.4499}  #, '1064': 18.4375}
bac_sim = {'355': 1.1590, '532': 1.0714, '1064': 0.3688}
depol_sim = {'355': 0.3132, '532': 0.3144, '1064': 0.2392}'''

# Taklamakan case, model='ih', aero_type='dust'
'''case_name = 'TaklamakanDust'
ext_sim = {'355': 161.14798547, '532': 159.64912126}
bac_sim = {'355': 3.22815417, '532': 3.56560646, '1064': 3.05100229}
depol_sim = {'355': 0.31825725, '532': 0.33410372, '1064': 0.30904789}'''

# Saharan case, model='ih', aero_type='dust'
case_name = 'SaharanDust'
ext_sim = {'355': 639.2, '532': 650}
bac_sim = {'355': 9.4, '532': 13, '1064': 12}
depol_sim = {'355': 0.25, '532': 0.34, '1064': 0.23}

# to run the algorithm, firstly specify the a priori configuration
model = 'ih'  # or: 'sphere', 'spheroid', 'ih'
aero_type = 'dust'  # or: 'bba', 'urban' or 'ss' (sea salt)

# run BOREAL
Retrieval, Fit = BOREAL.inversion(ext=ext_sim, bac=bac_sim, aero_type=aero_type, model=model, depol=depol_sim)
filepath_fit = BOREAL.plot_fit(fit=Fit, output_dir=output_dir, name=case_name+'fit_boreal')
filepath_rtv = BOREAL.plot_rtv(retrieval=Retrieval, output_dir=output_dir, name=case_name+'rtv_boreal')
filepath_txt = BOREAL.export_txt(Retrieval, Fit, product_output_dir=output_dir, filename=case_name+'results_boreal') #, site_info=[0], timestamp_str='2022,02,22')
print(filepath_fit)
print(filepath_rtv)
print(filepath_txt)

# run BOREAL-PC (currently, the valid inputs of BOREAL-PC are restricted to 2a+3b+3d optical data;
# and it is intended for dust retrieval, so 'model' and 'aero_type' are deactivated).
opt_meas = BOREAL_PC.harmonize_opt_format(ext_sim, bac_sim, depol_sim)
boreal_pc_instance = BOREAL_PC.Retrieval_bimodal(opt_meas=opt_meas)
boreal_pc_rtv = boreal_pc_instance.do_retrieval()
filepath_txt = BOREAL_PC.export_txt(boreal_pc_rtv, product_output_dir=output_dir, filename=case_name + '_boreal_pc')
print(filepath_txt)
