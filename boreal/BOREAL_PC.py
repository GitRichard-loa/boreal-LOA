# Retrieval of the state parameters: [Vt, rVc, Sgc, fvf, mR, mI355] from 3+2+3 lidar measurements

import os
import numpy as np
from scipy.optimize import least_squares
from .forward_module.tamuDUST import ih
#import forward_module.tamuDUST.ih as fwd_model
from pandas import DataFrame, Series

fwd_model = ih


def combined_SD(cov_x, grad_x):
    """
    calculate the combined uncertainty of a scalar y which is a function of multivariate x
    :param cov_x: covariance matrix of the vector x
    :param grad_x: gradient vector of y to x
    :return:
    """
    var_y = 0
    for ii in range(cov_x.shape[0]):
        for jj in range(cov_x.shape[1]):
            var_y += cov_x[ii, jj] * grad_x[0, ii] * grad_x[0, jj]
    return var_y**0.5


def derivative(func, xx, args=None, dx=1e-7):
    """
    calculate the Jacobian matrix and the value vector of the vector-function 'f' at 'x'
    :param args: other parameters in f(x), a dictionary
    :param func: objective function, callable, output a 1-D array as the objective vector
    :param xx: the variety vector of 'f', 1-D array (ndarray or list)
    :param dx: delta
    :return: J(x)-ndarray and f(x)-ndarray
    """
    if args is None:
        args = {}
    xx = list(xx)
    y = func(xx, **args)
    try:
        Jacob_T = np.empty((len(xx), len(y)))
    except TypeError:
        Jacob_T = np.empty((len(xx), 1))
    for line_num in range(Jacob_T.shape[0]):
        x_new = xx[:]
        x_new[line_num] += dx
        y_new = func(x_new, **args)
        Jacob_T[line_num] = (y_new - y) / dx
    Jacob = Jacob_T.T
    return Jacob, y


def export_txt(boreal_pc_rtv, product_output_dir, filename, site_info=None, timestamp_str=None):
    if product_output_dir is not None:
        filepath = os.path.join(product_output_dir, filename + '_output.txt')

        with open(filepath, 'w') as f:

            if site_info is not None:
                for ele in site_info:
                    f.write("%s = %s\n" % (ele, str(site_info[ele])))

            if timestamp_str is not None:
                timestamp_str = timestamp_str.replace(',', '_')
                timestamp_str = timestamp_str.replace(':', '-')
                f.write("Average Time = %s\n" % timestamp_str)

            f.write('\n' + "*" * 20 + "   Retrieval message: %s   " % boreal_pc_rtv['rtv_message'] + "*" * 20 + '\n')
            f.write('Retrieved microphysical properties and SSA:\n')
        boreal_pc_rtv['state_params_rtv'].to_csv(filepath, mode='a')
        with open(filepath, 'a') as f:
            f.write('\nRetrieved volume size distribution:\n')
        boreal_pc_rtv['vsd_rtv'].to_csv(filepath, index_label='r (um)', mode='a')
        with open(filepath, 'a') as f:
            f.write('\nCovariance matrices of the logarithmic state parameters:\n')
        for key, longname in zip(['rtv_cov_tot', 'rtv_cov_se', 'rtv_cov_mn'],
                                 ['Total retrieval covariance', 'Retrieval covariance for smoothing error', 'Retrieval covariance for measurement noise']):
            f = open(filepath, 'a')
            f.write('\n' + longname + '\n')
            f.close()
            boreal_pc_rtv[key].to_csv(filepath, mode='a')
        with open(filepath, 'a') as f:
            f.write('\nOptical measurement fit:\n')
        boreal_pc_rtv['opt_fit'].to_csv(filepath, mode='a')
        with open(filepath, 'a') as f:
            f.write('\nrmse_fit (%%): %.2f' % boreal_pc_rtv['rmse_fit'])
    else:
        filepath = filename + '_output.txt'
    return filepath


def from_Lognormal_to_normalLog(mean, std):
    """
    convert the mean and std of a lognormal distribution in linear domain to the mean and std of its logarithm in the log domain
    :param mean:
    :param std:
    :return:
    """
    rel_std = std / mean
    std_in_log = np.log(rel_std**2 + 1)**0.5
    mean_in_log = 0.5 * np.log(mean**2 / (rel_std**2 + 1))
    return mean_in_log, std_in_log


def fwd_func(ln_state):
    """
    forward function to calculate optical properties from the log of the state x
    :param ln_state: log of 'Vt', 'rVc', 'Sgc', 'fvf', 'mR', 'mI355'
    :return:
    """
    x = np.exp(ln_state)
    vsd = lognormal_sd(radius_grid, rVf, x[3]*x[0], Sgf) + lognormal_sd(radius_grid, x[1], (1-x[3])*x[0], x[2])
    y = fwd_model.get_opt(np.concatenate((vsd, x[4:])), radius_grid, wave_ext=waves[:2], wave_bac=waves, wave_depol=waves, only_fwd=False, mI_dusttype=True)
    return np.log(y)


def get_covrtv_degnonlinear_avemat(cov_mm, cov_a, jac_mat):
    """
    calculate retrieval covariance matrix, degree of nonlinearity and averaging matrix for a specific state
    :return:
    """
    cov_a_I = np.linalg.inv(cov_a)
    cov_m_I = np.linalg.inv(cov_mm)
    KTSmIK = np.matmul(jac_mat.T, np.matmul(cov_m_I, jac_mat))
    cov_rtv = np.linalg.inv(KTSmIK + cov_a_I)
    cov_rtv_noise = np.matmul(cov_rtv, np.matmul(KTSmIK, cov_rtv))
    cov_rtv_a = cov_rtv - cov_rtv_noise
    A_matrix = np.matmul(cov_rtv, KTSmIK)
    return cov_rtv, cov_rtv_a, cov_rtv_noise, A_matrix


def get_integ_params(r_grid, dv_lnr):
    """
    calculate vt, r_mean, Sg, and reff for a given dv_lnr with r_grid
    :param r_grid:
    :param dv_lnr:
    :return:
    """
    lnr_grid = np.log(r_grid)
    vt = np.trapz(dv_lnr, lnr_grid)
    reff = vt / np.trapz(dv_lnr/r_grid, lnr_grid)
    lnr_mean = np.trapz(y=lnr_grid * dv_lnr, x=lnr_grid) / vt
    r_mean = np.exp(lnr_mean)
    lnSg = (np.trapz((lnr_grid - lnr_mean) ** 2 * dv_lnr, lnr_grid) / vt)**0.5
    Sg = np.exp(lnSg)  # geometric standard deviation
    return vt, r_mean, Sg, reff


def get_ssa(ln_state):
    """
    calculate spectral ssa from the ln_state
    :param ln_state: log of Vt, rVc, Sgc, fvf, mR, mI355
    :return:
    """
    state_params = np.exp(ln_state)
    dV_dlnr = lognormal_sd(radius_grid, rVf, state_params[0] * state_params[3], Sgf) + lognormal_sd(radius_grid, state_params[1], (1 - state_params[3]) * state_params[0], state_params[2])
    opt_cal = fwd_model.get_opt(np.concatenate((dV_dlnr, state_params[-2:])), radius_grid, only_fwd=True, mI_dusttype=True)
    return opt_cal[3]


def harmonize_opt_format(ext_dict, bac_dict, depol_dict):
    """
    convert the format of input optical data to the standard input of BOREAL-PC
    :param ext_dict: dict, extinction at 355 and 532 nm
    :param bac_dict: dict, backscattering at 355, 532 and 1064 nm
    :param depol_dict: dict, depolarization ratio at 355, 532 and 1064 nm
    :return:
    """
    #meas_array = np.array([ext_dict['355'], ext_dict['532'], bac_dict['355'], bac_dict['532'], bac_dict['1064'],
                          #depol_dict['355'], depol_dict['532'], depol_dict['1064']])
    meas_dict = {}
    for keys, dicts in zip(['ext_', 'bac_', 'depol_'], [ext_dict, bac_dict, depol_dict]):
        for k, item in dicts.items():
            meas_dict[keys+k] = item
    return Series(meas_dict)


def lognormal_sd(r_grid, r_mod, C, gsd):
    """
    get lognormal size distribution (dC/dlnr) for a set of radius grids
    :param r_grid: 1D-ndarray, radius grids in um
    :param r_mod: float, median or mode radius
    :param C: float, concentration
    :param gsd: float, geometry standard deviation
    :return: array of the size distribution having the same size as r_grid
    """
    ln_Sg = np.log(gsd)
    return C * np.exp(-(np.log(r_grid) - np.log(r_mod)) ** 2 / (2 * ln_Sg ** 2)) / ((2 * np.pi) ** 0.5 * ln_Sg)


def reff_from_logstate(ln_state):
    """
    calculate effective radius from the log of the state vector.
    :param ln_state: logarithm of [rVc, Sgc, fvf]
    :return:
    """
    rVc, Sgc, fvf = np.exp(ln_state)
    M2 = np.array([rV**2 * np.exp(-4*np.log(Sg)**2) for rV, Sg in zip([rVf, rVc], [Sgf, Sgc])])
    M3 = np.array([rV**3 * np.exp(-4.5*np.log(Sg)**2) for rV, Sg in zip([rVf, rVc], [Sgf, Sgc])])
    Vt_weight = np.array([fvf, (1 - fvf)])
    return np.sum(M3 * Vt_weight) / np.sum(M2 * Vt_weight)


def resi_vec(ln_state_vector, measurement_vector, meas_ind, std_of_ln_meas_valid_data, apriori_value=None, apriori_std=None):
    """
    residual vector to minimize
    :param meas_ind: keys corresponding to the measurement_vector (to specify the order of the opt params)
    :param std_of_ln_meas_valid_data: pure data vector of the std_of_ln_meas Series
    :param ln_state_vector: ln[Vt, rVc, Sgc, fvf, mR, mI355]
    #:param measurement_vector: 2a+3b+3d lidar measurements
    :param measurement_vector: pure data vector from the measurement Series
    :param apriori_value: if incorporating a priori value
    :param apriori_std: if incorporation a priori standard deviation
    :return:
    """
    ln_opt_cal = fwd_func(ln_state_vector)
    ln_opt_cal_series = Series(ln_opt_cal, index=opt_params_index_full)
    ln_opt_cal_valid_data = ln_opt_cal_series[meas_ind].values
    resi_meas = (np.log(measurement_vector) - ln_opt_cal_valid_data) / std_of_ln_meas_valid_data
    if apriori_value is not None:
        resi_apriori = (ln_state_vector - apriori_value) / apriori_std
        residual_vec = np.concatenate((resi_meas, resi_apriori))
    else:
        residual_vec = resi_meas
    return residual_vec


# configuration:
rVf, Sgf = 0.15, 1.9
waves = [0.355, 0.532, 1.064]
radius_grid = fwd_model.r_grid_org
vt_std = 0.26  # according to the fitting of the in situ measurements
mR_a, mI355_a = 1.518, 0.003
mR_std, mI355_std = 0.022, 0.002
ln_state_param_names = ['ln(Vt)', 'ln(Rc)', 'ln(Sc)', 'ln(fvf)', 'ln(mR)', 'ln(mI355)']

# measurement error
opt_params_index_full = ['ext_355', 'ext_532', 'bac_355', 'bac_532', 'bac_1064', 'depol_355', 'depol_532', 'depol_1064']
max_meas_err = Series([0.1, 0.1, 0.1, 0.1, 0.2, 0.15, 0.15, 0.15], index=opt_params_index_full)
rel_std_of_meas = max_meas_err / 3
std_of_ln_meas = (np.log(rel_std_of_meas**2 + 1))**0.5


class Retrieval_bimodal:
    def __init__(self, opt_meas):  # opt_meas is a pd.Series with keys in the order 'ext_wave', 'bac_wave', 'depol_wave'
        self.opt_meas_data = opt_meas.values
        self.opt_meas_ind = opt_meas.index
        # determine the a priori constriant on vt using the linear regression result with ext_532
        vt_a = 0.89 * opt_meas['ext_532']
        rV_a, Sg_a = 2.97, 1.96
        rV_std, Sg_std = 1.12, 0.34
        fvf_a, fvf_std = 0.028, 0.045
        self.state_a, self.state_std_a = from_Lognormal_to_normalLog(np.array([vt_a, rV_a, Sg_a, fvf_a, mR_a, mI355_a]), np.array([vt_std * vt_a, rV_std, Sg_std, fvf_std, mR_std, mI355_std]))
        self.bounds = ([-np.inf, -np.inf, self.state_a[2] - 3*self.state_std_a[2], self.state_a[3] - 3*self.state_std_a[3], np.log(1.4), np.log(0.0001)], [np.inf, np.inf, self.state_a[2] + 3*self.state_std_a[2], self.state_a[3] + 3*self.state_std_a[3], np.log(1.68), np.log(0.06)])
        self.std_of_ln_meas = std_of_ln_meas[self.opt_meas_ind]
        self.std_of_ln_meas_data = self.std_of_ln_meas.values
        self.cov_m = np.diag(self.std_of_ln_meas_data**2)

    def do_retrieval(self, apriori_constraint=True):
        if apriori_constraint:
            args = [self.opt_meas_data, self.opt_meas_ind, self.std_of_ln_meas, self.state_a, self.state_std_a]
        else:
            args = [self.opt_meas_data, self.opt_meas_ind, self.std_of_ln_meas]
        res = least_squares(resi_vec, self.state_a, bounds=self.bounds, args=args)
        state_rtv = np.exp(res.x)
        rtv_message = res.message
        vsd_rtv = lognormal_sd(radius_grid, rVf, state_rtv[3] * state_rtv[0], Sgf) + \
                  lognormal_sd(radius_grid, state_rtv[1], (1 - state_rtv[3]) * state_rtv[0], state_rtv[2])
        grad_reff_to_lnstate, reff_rtv = derivative(reff_from_logstate, res.x[1:4])
        jac_mat_ssa_to_lnstate, ssa_rtv = derivative(get_ssa, res.x)
        # retrieval covariance and combined uncertainty
        jac_mat_lnopt_to_lnstate, ln_opt = derivative(fwd_func, res.x)
        jac_mat_lnopt_to_lnstate_df = DataFrame(jac_mat_lnopt_to_lnstate, index=opt_params_index_full)
        jac_mat_lnopt_to_lnstate_matter = jac_mat_lnopt_to_lnstate_df.loc[self.opt_meas_ind].values
        cov_a = np.diag(self.state_std_a**2)
        cov_rtv, cov_rtv_smooth_err, cov_rtv_meas_noise, A_mat = get_covrtv_degnonlinear_avemat(self.cov_m, cov_a, jac_mat_lnopt_to_lnstate_matter)
        # standard deviations (absolute) of the retrieved states in the linear space: total std, smoothing error std, meas noise std
        std_rtv_linearspace, reff_std, ssa_std = [], [], []
        for cov in [cov_rtv, cov_rtv_smooth_err, cov_rtv_meas_noise]:
            std_rtv_linearspace.append(((np.exp(np.diag(cov)) - 1) * state_rtv ** 2) ** 0.5)
            reff_std.append(combined_SD(cov[1:4, 1:4], grad_reff_to_lnstate))
            ssa_std.append([combined_SD(cov, jac_mat_ssa_to_lnstate[i, np.newaxis]) for i in range(jac_mat_ssa_to_lnstate.shape[0])])
        # fitting error and recalculated optical data
        residual_vector = res.fun[:len(self.opt_meas_data)]
        opt_recal = self.opt_meas_data / np.exp(residual_vector * self.std_of_ln_meas_data)
        opt_diff = (np.exp(-(residual_vector * self.std_of_ln_meas_data)) - 1) * 100
        fit_rmse = np.linalg.norm(opt_diff) / len(opt_diff) ** 0.5

        # organize the output
        rtv_sd_all = np.concatenate((std_rtv_linearspace, np.array([reff_std]).T, ssa_std), axis=1)
        rtv_result_all = np.concatenate((state_rtv, [reff_rtv], ssa_rtv))
        rtv_all_df = DataFrame(np.vstack((rtv_result_all, rtv_sd_all)).T, index=['Vt', 'Rc', 'Sc', 'fvf', 'mR', 'mI355', 'reff', 'SSA355', 'SSA532', 'SSA1064'],
                               columns=['rtv_value', 'rtv_sd_tot', 'rtv_sd_se', 'rtv_sd_mn'])
        vsd_rtv_series = Series(vsd_rtv, index=radius_grid.round(4).astype(str), name='dV/dlnr (um^3/cm^3)')
        opt_fit_df = DataFrame(np.vstack((self.opt_meas_data, opt_recal, opt_diff)).T, columns=['Opt. measured', 'Opt. fitted', 'Fitting error (%)'],
                               index=self.opt_meas_ind)

        cov_rtv_df = [DataFrame(cov, columns=ln_state_param_names, index=ln_state_param_names) for cov in [cov_rtv, cov_rtv_smooth_err, cov_rtv_meas_noise]]

        return {'rtv_message': rtv_message, 'state_params_rtv': rtv_all_df, 'vsd_rtv': vsd_rtv_series, 'opt_fit': opt_fit_df,
                'rmse_fit': fit_rmse, 'rtv_cov_tot': cov_rtv_df[0], 'rtv_cov_se': cov_rtv_df[1], 'rtv_cov_mn': cov_rtv_df[2]}
