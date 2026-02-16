#!/usr/bin/env python3

import os
import numpy as np
import math
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
from datetime import datetime, timezone
from matplotlib.font_manager import FontProperties
from configparser import ConfigParser
from .forward_module.spheroid import spheroid
from .forward_module.tamuDUST import ih

# current_dir = os.path.dirname(os.path.abspath(__file__))


def get_BOREAL_version(setup_dir):
    setup_file = os.path.join(setup_dir, 'setup.cfg')
    config = ConfigParser()
    config.read(setup_file)
    return config['metadata']['version']


#setup_directory = os.path.join(current_dir, '..')
#BOREAL_v = get_BOREAL_version(setup_directory)


class BorealException(Exception):
    pass


# construct the second difference operator, length of the objective vector - 2 !!!
def construct_U2(length):
    k_array = np.zeros((length - 2, length))
    for h_i in range(length - 2):
        k_array[h_i, h_i] = 1
        k_array[h_i, h_i + 1] = -2
        k_array[h_i, h_i + 2] = 1
    zero = np.zeros((length - 2, 2))
    array_jacob = np.concatenate((k_array, zero), axis=1)
    return k_array, array_jacob


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


# get the number of modes of a particle size distribution
def get_mode_num(psd):
    # locate local maxima
    diff_v = psd[1:] - psd[: -1]
    vmax_num = 0  # Note: they are the indices of local maxima
    for ii in range(1, len(psd) - 1):
        if (diff_v[ii] < 0) & (diff_v[ii - 1] > 0):
            vmax_num += 1
    return vmax_num


def Vt_Reff(grid, vi):
    """
    calculate volume concentration, effective radius from a VSD
    :return:
    """
    ln_grid = np.log(grid)
    concentrate_vol = np.trapz(vi, ln_grid)
    effective_r = concentrate_vol / np.trapz(vi / grid, ln_grid)
    return concentrate_vol, effective_r


def statistic_quant(data):
    """
    get mean and std of an array. If the array is 2d of size (m, n), the statistics is conducted
    along the m-dimension, resulting in a 1d array of size n.
    :param data: numpy or list data
    :return:
    """
    nd_data = np.array(data)
    if len(nd_data.shape) == 1:
        data_mean = np.mean(nd_data)
        data_std = np.std(nd_data)
    else:
        data_mean = np.mean(nd_data, axis=0)
        data_std = np.std(nd_data, axis=0)
    return data_mean, data_std


def get_default_confg(aerosol_type):
    """
    get default configuration parameters for inversion()
    :return:
    """
    # input configuration
    na, sigma_na = 1.5, 0.1
    ka_dict = {'absorbing': 0.015, 'dust': 0.005, 'non-absorbing': 0.005}  # 'ss' for sea salt
    sigma_ka_dict = {'absorbing': 0.01, 'dust': 0.005, 'non-absorbing': 0.005}
    if aerosol_type == 'bba':
        ka = ka_dict['absorbing']
        sigma_ka = sigma_ka_dict['absorbing']
    elif aerosol_type == 'dust':
        ka = ka_dict['dust']
        sigma_ka = sigma_ka_dict['dust']
    else:
        ka = ka_dict['non-absorbing']
        sigma_ka = sigma_ka_dict['non-absorbing']
    rmin_min = 0.05
    rmin_max = 0.3
    rmax_min = 1.5
    rmax_max = 15
    r_regrid_num = 8

    epsilon2 = 2.5
    # epsilon3 = sigma_na / na
    epsilon_delta_x = 2.54  # (ln(xmax) - ln(xmin)) / 2
    epsilon_delta_n = 0.07  # (ln1.6 - ln1.4) / 2
    epsilon_delta_k = 2.3  # (ln0.1 - ln0.001) / 2 # scaling matrix

    return {'na': na, 'sigma_na': sigma_na, 'ka': ka, 'sigma_ka': sigma_ka, 'rmin_min': rmin_min, 'rmin_max': rmin_max,
            'rmax_min': rmax_min, 'rmax_max': rmax_max, 'r_regrid_num': r_regrid_num, 'epsilon_delta_x': epsilon_delta_x,
            'epsilon_delta_n': epsilon_delta_n, 'epsilon_delta_k': epsilon_delta_k, 'epsilon2': epsilon2}


def get_default_measerr():
    """
    get default measurement error
    :return: a dict
    """
    return {'ext_err': 0.1, 'bac_err': 0.1, 'depol_err': 0.1, 'lr_err': 0.15}


def __do_iter(beta, Y, fwd_func, arg_fwd, D, U, J2, J3, J4, SI, r_regrid_num, res_vec):
    """
    do iteration
    :param Y: list, ln[y1, y2, y3, y4]
    :param fwd_func: callable, function for forward calculation, called by derivative()
    :param beta: list or ndarray, logarithm of the initial value of the state parameter vector
    :param res_vec: ndarray, measurement error vector, relative error. The size is equal to Y[0].
    :param arg_fwd: dict, arguments of fwd_model.get_opt other than the state parameters
    :return:
    """
    # the first iteration, i.e., initialization
    Iter_num = 0
    beta_exp = np.exp(beta)
    # calculate the Jacobian matrix
    J1, f1 = derivative(func=fwd_func, xx=beta_exp, args=arg_fwd)  # array
    U1 = J1 * beta_exp / f1[:, np.newaxis]
    # calculate fl
    Xi = np.mat(beta[:-2]).T
    F1 = np.log(f1)
    F2 = np.array(U * Xi).ravel()
    F3 = beta[-2]  # f3 = ni
    F4 = beta[-1]  # f4 = ki
    Fl = [F1, F2, F3, F4]
    J = [np.mat(U1), np.mat(J2), np.mat(J3), np.mat(J4)]
    num_gener_meas = F1.size + F2.size + 2
    c = (num_gener_meas - r_regrid_num - 2)
    sum3 = 0
    for l in range(len(Fl)):
        deltaY_l = np.mat(Y[l] - Fl[l]).T
        sum3 += deltaY_l.T * SI[l] * deltaY_l
    chi_square = sum3[0, 0]
    damp_fac = 2 / c * chi_square
    enlarge_factor = 2

    # the following iterations
    while Iter_num <= 30:
        sum1, sum2 = 0, 0
        for l in range(len(Fl)):
            alpha_l = J[l].T * SI[l]
            deltaY_l = np.mat(Y[l] - Fl[l]).T
            sum1 += alpha_l * J[l]
            sum2 += alpha_l * deltaY_l
        G = sum1 + damp_fac * D
        b = sum2
        GI = G.I
        delta_beta = np.array(GI * b).ravel()

        beta_after = delta_beta + beta
        beta_exp_after = np.exp(beta_after)
        # calculate the Jacobian matrix
        J1_after, f1_after = derivative(func=fwd_func, xx=beta_exp_after, args=arg_fwd)  # array
        U1_after = J1_after * beta_exp_after / f1_after[:, np.newaxis]
        # calculate fl
        Xi_after = np.mat(beta_after[:-2]).T
        F1_after = np.log(f1_after)
        F2_after = np.array(U * Xi_after).ravel()
        F3_after = beta_after[-2]  # f3 = ni
        F4_after = beta_after[-1]  # f4 = ki
        Fl_after = [F1_after, F2_after, F3_after, F4_after]
        J_after = [np.mat(U1_after), np.mat(J2), np.mat(J3), np.mat(J4)]
        sum3_after = 0
        for l in range(len(Fl_after)):
            deltaY_l = np.mat(Y[l] - Fl_after[l]).T
            sum3_after += deltaY_l.T * SI[l] * deltaY_l
        chi_square_after = sum3_after[0, 0]
        Iter_num += 1
        if chi_square_after >= chi_square:
            damp_fac *= enlarge_factor
            enlarge_factor *= 2
        else:
            chi_square = chi_square_after
            damp_fac = max(2 / c * chi_square, damp_fac / 3)
            beta = beta_after
            Fl = Fl_after
            J = J_after
            enlarge_factor = 2

        deltaY_1 = np.mat(Y[0] - Fl_after[0]).T
        delta_Y1_array = np.array(deltaY_1).ravel()
        rel_err = np.fabs(np.exp(-delta_Y1_array) - 1)
        # stop = (Iter_num >= 30) | (np.all(res_vec - rel_err) & (chi_square_after < c))
        stop = np.all(res_vec - rel_err) & (chi_square < c)
        if stop:
            break

    e_for_measurement = np.linalg.norm(rel_err) / rel_err.size ** 0.5 * 100
    return beta, Iter_num, Fl[0], chi_square, e_for_measurement


def filt_inval_meas(name, dict_data):
    """
    filter invalid measurements (NaN, float less than zero are considered as 'invalid')
    :param dict_data:
    :param name:
    :return: a Series type if the input data are valid
    """
    for dict_item in dict_data.items():
        if dict_item[1] < 0:
            raise BorealException('%s at %s nm is negative.' % (name, dict_item[0]))
    seri_data = Series(dict_data)
    return seri_data


def check_meas_and_err(name, meas, err, err_default):
    if not len(meas):
        raise BorealException('%s measurements cannot be empty.' % name)
    meas_series = filt_inval_meas(name, meas)
    if (err is None) or (not len(err)):
        err = err_default
    err_series = filt_inval_meas(name+'_err', err)
    if len(set(err_series.index) & set(meas_series.index)):
        err_series = err_series[meas_series.index]
    else:
        raise BorealException('For %s: the error wavelengths is not consistent with the measurement wavelengths.' % name)
    meas_wave = meas_series.index.values.astype(float) * 0.001
    meas_series.index = meas_wave
    err_series.index = meas_wave
    meas_wave.sort()

    return meas_series.reindex(meas_wave), err_series.reindex(meas_wave), meas_wave


def inversion(ext, bac, aero_type, model, depol=None, ext_err=None, bac_err=None, depol_err=None, config=None):
    """
    main function to implement the inverse procedure
    :param ext: dict, spectral ext. coef., the keys (str) are wavelength in nm, the values are corresponding measurements (float) in 1/Mm
    :param bac: dict, spectral bac. coef., the keys (str) are wavelength in nm, the values are corresponding measurements (float) in 1/(Mm*sr)
    :param depol: dict or None (default=None), spectral depol., the keys (str) are wavelength in nm, the values are corresponding measurements (float) (unit of 1)
    :param aero_type: str, 'dust', 'bba', 'urban' or 'ss' (sea salt)
    :param model: str, 'sphere', 'spheroid' or 'ih', forward model (scattering model) used in the inversion
    :param ext_err: dict or None (default=None), maximum measurement error in ext (three times of measurement std). None for default values.
    :param bac_err: dict or None, measurement error in bac, having the same form with bac. If None, get default configuration. By default, None
    :param depol_err: dict or None, measurement error in depol, having the same form with depol. If None, get default configuration. By default, None
    :param config: dict, other configuration parameters
    :return: two dicts: Retrieval and Fit
    """
    # time info:
    now_utc = datetime.now(tz=timezone.utc)
    now_local = now_utc.astimezone()  # covert timezone to the system default.
    strfmd_now_utc = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
    strfmd_now_local = now_local.strftime('%Y-%m-%dT%H:%M:%S')
    str_time = "Processed at:\n%s (%s time)\n%s (%s time)" % (strfmd_now_utc, now_utc.tzname(), strfmd_now_local, now_local.tzname())

    # pre-treatment
    # check model name and aerosol type
    if aero_type not in ['dust', 'bba', 'urban', 'ss']:
        raise BorealException('The aerosol type must be one of the following: %s' % 'absorbing, non-absorbing, dust')
    if aero_type == 'dust':
        mI_dusttype = True
    else:
        mI_dusttype = False

    if model == 'ih':
        r_grid_org = ih.r_grid_org
        get_opt = ih.get_opt
    elif (model == 'spheroid') | (model == 'sphere'):
        r_grid_org = spheroid.r_grid_org
        get_opt = spheroid.get_opt
    else:
        raise BorealException('The model name must be one of the following: %s' % 'sphere, spheroid, ih')

    # check ext & ext_err; bac & bac_err
    err_default_ext_and_bac = {'355': 0.1, '532': 0.1, '1064': 0.2}
    err_default_depol = {'355': 0.15, '532': 0.15, '1064': 0.15}
    ext_series, ext_err_series, ext_wave = check_meas_and_err('Extinction', ext, ext_err, err_default=err_default_ext_and_bac)
    bac_series, bac_err_series, bac_wave = check_meas_and_err('Backscattering', bac, bac_err, err_default=err_default_ext_and_bac)

    # check depol
    # check availability of depol meas
    if (depol is None) or (not len(depol)):
        depol_wave = None
        depol_series = {}
        depol_err_series = {}
    else:
        depol_series, depol_err_series, depol_wave = check_meas_and_err('Depolarization', depol, depol_err, err_default=err_default_depol)

    # determine inversion dataset
    if (model == 'sphere') | (depol_wave is None):
        y1 = np.concatenate((ext_series.values, bac_series.values))
        epsilon1 = np.concatenate((ext_err_series.values, bac_err_series.values)) / 3
        print('The %s model is used to invert alpha at %s um, and beta at %s um' % (model, ', '.join([str(i) for i in ext_series.index]), ', '.join([str(i) for i in bac_series.index])))
    else:
        y1 = np.concatenate((ext_series.values, bac_series.values, depol_series.values))
        epsilon1 = np.concatenate((ext_err_series.values, bac_err_series.values, depol_err_series.values)) / 3
        print('The %s model is used to invert alpha at %s um, beta at %s um, and delta at %s um' % (
              model, ', '.join([str(i) for i in ext_series.index]), ', '.join([str(i) for i in bac_series.index]), ', '.join([str(i) for i in depol_series.index])))

    args0 = {'wave_ext': ext_wave, 'wave_bac': bac_wave, 'wave_depol': depol_wave, 'mI_dusttype': mI_dusttype, 'model': model, 'only_fwd': False}

    # check if the length of y1 >= 3, otherwise, c <= 0 (underdetermined system, measurements < state parameters)
    if len(y1) < 3:
        raise BorealException('Underdetermined inverse system: the total number of the input measurements is less than 3')

    # calculate lidar ratio
    lr_series = ext_series / bac_series

    # calculate lidar ratio error (maximum relative error)
    partial_lr_ext = 1 / bac_series
    partial_lr_bac = - ext_series / bac_series ** 2
    lr_err_term1 = (ext_err_series / lr_series * partial_lr_ext * ext_err_series / 3) ** 2
    lr_err_term2 = (bac_series / lr_series * partial_lr_bac * bac_err_series / 3) ** 2
    lr_err_series = (lr_err_term1 + lr_err_term2) ** 0.5 * 3
    lr_series = lr_series.dropna()
    lr_err_series = lr_err_series.dropna()

    if config is None:
        config = get_default_confg(aerosol_type=aero_type)

    rmin_start = np.argmin(np.fabs(r_grid_org - config['rmin_min']))
    rmin_end = np.argmin(np.fabs(r_grid_org - config['rmin_max']))
    rmax_start = np.argmin(np.fabs(r_grid_org - config['rmax_min']))
    rmax_end = np.argmin(np.fabs(r_grid_org - config['rmax_max']))
    ka, sigma_ka = config['ka'], config['sigma_ka']

    D = np.mat(np.diag(np.hstack((np.ones(config['r_regrid_num']) / config['epsilon_delta_x'] ** 2,
                                  1 / config['epsilon_delta_n'] ** 2, 1 / config['epsilon_delta_k'] ** 2))))
    U, J2 = construct_U2(length=config['r_regrid_num'])  # array
    J3 = np.hstack((np.zeros(config['r_regrid_num']), [1, 0]))
    J4 = np.hstack((np.zeros(config['r_regrid_num']), [0, 1]))
    Y2 = np.zeros(config['r_regrid_num'] - 2)

    S2 = np.diag(config['epsilon2'] ** 2 * np.ones(config['r_regrid_num'] - 2))
    epsilon3 = config['sigma_na'] / config['na']
    S3 = np.diag(epsilon3 ** 2 * np.ones(1))  # na cov
    config2 = {'D': D, 'U': U, 'J2': J2, 'J3': J3, 'J4': J4}

    r_grid_eff = r_grid_org[rmin_start: rmax_end + 1]
    ln_r_grid_eff = np.log(r_grid_eff)  # both the kernel function and dV_dlnr should be interpolated in the log-scale
    epsilon4 = math.log((1 + (1 + 4 * (sigma_ka / ka) ** 2) ** 0.5) / 2) ** 0.5
    S4 = np.diag(epsilon4 ** 2 * np.ones(1))  # ka cov
    Y = [np.log(y1), Y2, math.log(config['na']), math.log(ka)]
    rms_epsilon1 = np.linalg.norm(epsilon1) / epsilon1.size ** 0.5 * 100
    config2['res_vec'] = epsilon1 * 3
    scale_fac_for_epsilon1 = 1

    while True:
        epsilon1_enlarge = scale_fac_for_epsilon1 * epsilon1
        S1 = np.diag(epsilon1_enlarge ** 2)  # measurement cov
        SI = [np.mat(S1).I, np.mat(S2).I, np.mat(S3).I, np.mat(S4).I]
        config2['SI'] = SI

        # inversion for all the inversion windows
        Solution_org = []
        Solution_org1 = []
        VSD_IS = []  # list of individual solutions before the selection criteria are applied
        for rmin_index in range(rmin_start, rmin_end + 1):
            for rmax_index in range(rmax_start, rmax_end + 1):
                if rmax_index - rmin_index < 14:  # #####################
                    continue
                r_grid_cutforinv = r_grid_org[rmin_index: rmax_index + 1]
                psd_unif = np.ones(config['r_regrid_num'])
                ln_r_grid_inv = np.linspace(math.log(r_grid_cutforinv[0]), math.log(r_grid_cutforinv[-1]), num=config['r_regrid_num'])
                r_grid_inv = np.exp(ln_r_grid_inv)
                args1 = {'r_grid': r_grid_cutforinv, 'lnr_grid_redu': ln_r_grid_inv}
                args = {**args0, **args1}
                # initialize VSD as a uniform distribution calculated from ext532 (or ext355)
                ext_unif = get_opt(np.hstack((psd_unif, config['na'], ka)), **args)[: len(ext_wave)]
                for wave_unif, ind_in_ext_wave in zip([0.532, 0.355, 1.064], [1, 0, 2]):
                    if wave_unif in ext_wave:
                        VSD_initial = 0.05 * ext_series[wave_unif] / ext_unif[ind_in_ext_wave] * np.ones(config['r_regrid_num'])
                        break

                State_param0 = np.log(np.hstack((VSD_initial, config['na'], ka)))  # the reason why don't put it outside the loop is that the iteration in "__do_iter" will change its value.
                # Thus it needs initialisation each time.
                try:
                    log_xi, iter_num, Fl_after0, cost_func, e_for_measurement = __do_iter(State_param0, Y, fwd_func=get_opt, r_regrid_num=config['r_regrid_num'], arg_fwd=args, **config2)
                except (IndexError, ValueError):
                    continue
                xi = np.exp(log_xi)
                # y1_recal = np.exp(-delta_Y1_array + self.Y[0])
                # first filter: threshold of geometric variance of the retrieved size distribution
                vsd_is = xi[:-2]
                VSD_IS.append(np.vstack((r_grid_inv, vsd_is)))  # collect every individual solution (r_grid_inv, vsd_is)
                vi_max = vsd_is.max()
                threshold1 = 0.5 * vi_max
                # threshold1 = 0.7 * vi_max  # historical used
                threshold2 = 0.05 * vi_max
                args['r_grid_inv'] = r_grid_inv
                if (((vsd_is[0] < vsd_is[1]) & (vsd_is[0] <= threshold1)) | ((vsd_is[0] >= vsd_is[1]) & (vsd_is[0] <= threshold2))) & \
                        (((vsd_is[-1] < vsd_is[-2]) & (vsd_is[-1] <= threshold1)) | ((vsd_is[-1] >= vsd_is[-2]) & (vsd_is[-1] <= threshold2))):
                    mode_params = get_mode_num(vsd_is)
                    if mode_params <= 2:
                        Solution_org.append((e_for_measurement, args, xi, (rmin_index, rmax_index)))
                    else:
                        Solution_org1.append((e_for_measurement, args, xi, (rmin_index, rmax_index)))
                else:
                    Solution_org1.append((e_for_measurement, args, xi, (rmin_index, rmax_index)))
        if len(Solution_org1) or len(Solution_org):
            break
        elif scale_fac_for_epsilon1 < 4:
            scale_fac_for_epsilon1 += 1
            # print('Warning: Overflow happened during the iteration. Enlarge epsilon1 by a factor of %d and try again' % scale_fac_for_epsilon1)
        else:
            rtv_message = 'Warning: Overflow still happened even if epsilon1 was enlarged by a factor of %d. Not any individual solutions are found' % scale_fac_for_epsilon1
            print(rtv_message)
            rtv_flag = 0
            retrieval = {'rtv_message': rtv_message, 'rtv_flag': rtv_flag}
            fit = {}
            return retrieval, fit

    if len(Solution_org):
        # modification on 6 April: sort the discrepancy instead of using np.percentile
        Solution_org.sort()
        len_Solution = len(Solution_org)
        Solution_array = np.array(Solution_org, dtype=object)
        rtv_message = 'Individual solutions satisfying the constraints on modes and fringe size bins are found'
        rtv_flag = 1
        print(rtv_message)
    else:
        Solution_org1.sort()
        len_Solution = len(Solution_org1)
        Solution_array = np.array(Solution_org1, dtype=object)
        rtv_message = 'Warning: individual solutions are found, but none satisfies the constraints on modes or fringe size bins'
        rtv_flag = 2
        print(rtv_message)

    if len_Solution == 1:
        Solution_cut = Solution_array
    else:
        if len_Solution < 5:
            select_num = 1
        else:
            select_frac = 0.2
            select_num = int(len_Solution * select_frac)
        # bool_index = np.hstack((np.ones(1, dtype=int), np.zeros(len_Solution_org - 1, dtype=int)))  # Note!! The dtype should be 'bool' rather than 'int', because 'bool' indexing is in fact a 'fancy' indexing where the rows whose index is True will be obtained.
        # if the type of the indexing list is 'int', only copies of row 0 and row 1 are obtained!! In this case, np.nonzero() must be used to indicate the indices of '1' elements. See the example below.
        bool_index = np.hstack((np.ones(select_num, dtype=bool), np.zeros(len_Solution - select_num, dtype=bool)))
        # Solution_cut = Solution_array[np.nonzero(bool_index | (Solution_array[:, 0] < rms_epsilon1))]  # example of the use of np.nonzero() if the type of bool_index is 'int'
        Solution_cut = Solution_array[bool_index | (Solution_array[:, 0] < rms_epsilon1)]
    Num_of_solutions = Solution_cut.shape[0]
    vsd_coll, n_coll, k_coll, vt_coll, reff_coll = [], [], [], [], []
    ext_full_cal_coll, bac_full_cal_coll, ssa_full_coll, depol_full_cal_coll, lr_full_cal_coll = [], [], [], [], []
    residual_coll = []
    for i in range(Solution_cut.shape[0]):
        residual_coll.append(Solution_cut[i, 0])
        Solution_args = Solution_cut[i, 1]
        # vsd = np.interp(r_grid_eff, Solution_cut[i, 1], Solution_cut[i, 2][:-2], left=0, right=0)
        vsd = np.interp(ln_r_grid_eff, np.log(Solution_args['r_grid_inv']), Solution_cut[i, 2][:-2], left=0,  right=0)  # both the kernel function and dV_dlnr should be interpolated in the log-scale
        vsd_coll.append(vsd)
        n_coll.append(Solution_cut[i, 2][-2])
        k_coll.append(Solution_cut[i, 2][-1])
        vt, reff = Vt_Reff(grid=Solution_args['r_grid_inv'], vi=Solution_cut[i, 2][:-2])
        vt_coll.append(vt)
        reff_coll.append(reff)
        Solution_args.pop('r_grid_inv')
        Solution_args['only_fwd'] = True
        wave_union, ext_cal, bac_cal, ssa_cal, lr_cal, depol_cal = get_opt(Solution_cut[i, 2], **Solution_args)
        for item, container in zip([ext_cal, bac_cal, ssa_cal, lr_cal, depol_cal],
                                   [ext_full_cal_coll, bac_full_cal_coll, ssa_full_coll, lr_full_cal_coll,
                                    depol_full_cal_coll]):
            container.append(item)
    vsd_mean, vsd_std = statistic_quant(vsd_coll)
    cri_r_mean, cri_r_std = statistic_quant(n_coll)
    cri_i_mean, cri_i_std = statistic_quant(k_coll)
    vt_mean, vt_std = statistic_quant(vt_coll)
    reff_mean, reff_std = statistic_quant(reff_coll)
    ext_full_cal_mean, ext_full_cal_std = statistic_quant(ext_full_cal_coll)
    bac_full_cal_mean, bac_full_cal_std = statistic_quant(bac_full_cal_coll)
    ssa_full_mean, ssa_full_std = statistic_quant(ssa_full_coll)
    depol_full_cal_mean, depol_full_cal_std = statistic_quant(depol_full_cal_coll)
    lr_full_cal_mean, lr_full_cal_std = statistic_quant(lr_full_cal_coll)
    residual_mean, residual_std = statistic_quant(residual_coll)
    vt_eff, reff_eff = Vt_Reff(r_grid_eff, vsd_mean)
    state_param = np.hstack((vsd_mean, cri_r_mean, cri_i_mean))
    #args_for_recal = {'x_array': state_param, 'r_grid': r_grid_eff, 'model': model}
    wave_union, ext_full_cal_eff, bac_full_cal_eff, ssa_full_eff, lr_full_cal_eff, depol_full_cal_eff = get_opt(state_param, r_grid_eff, only_fwd=True, model=model, mI_dusttype=mI_dusttype)
    #y1_cal_eff = get_opt(only_fwd=False, **args_for_recal)
    y1_cal_eff = np.concatenate((ext_full_cal_eff[:len(ext_wave)], bac_full_cal_eff[:len(bac_wave)]))
    if depol_wave is not None:
        y1_cal_eff = np.append(y1_cal_eff, depol_full_cal_eff[:len(depol_wave)])
    residual_eff = np.linalg.norm((y1_cal_eff - y1) / y1) * 100

    #vsd_mean = np.concatenate(([0], vsd_mean, [0]))
    #vsd_std = np.concatenate(([0], vsd_std, [0]))
    #r_grid_eff = r_grid_org[rmin_start - 1: rmax_end + 2]

    retrieval = {'rtv_message': rtv_message, 'rtv_flag': rtv_flag, 'str_time': str_time,
                 'fwd_model': model, 'NIS': Num_of_solutions, 'r_grid': r_grid_eff,
                 'VSD_IS_all': VSD_IS, 'VSD_IS_selected': vsd_coll,
                 'VSD_mean': vsd_mean, 'VSD_std': vsd_std,
                 'CRI_r_mean': cri_r_mean, 'CRI_r_std': cri_r_std,
                 'CRI_i_mean': cri_i_mean, 'CRI_i_std': cri_i_std,
                 'Vt_mean': vt_mean, 'Vt_std': vt_std, 'Vt_eff': vt_eff,
                 'Reff_mean': reff_mean, 'Reff_std': reff_std, 'Reff_eff': reff_eff,
                 'Wave_union': wave_union,
                 'SSA_full_mean': ssa_full_mean, 'SSA_full_std': ssa_full_std, 'SSA_full_eff': ssa_full_eff}
    fit = {'rtv_message': rtv_message, 'str_time': str_time,
           'Ext_meas': ext_series, 'Ext_meas_err': ext_err_series,
           'Bac_meas': bac_series, 'Bac_meas_err': bac_err_series,
           'Depol_meas': depol_series, 'Depol_err': depol_err_series,
           'LR_meas': lr_series, 'LR_err': lr_err_series, 'Wave_union': wave_union,
           'Residual_mean': residual_mean, 'Residual_std': residual_std, 'Residual_eff': residual_eff,
           'Ext_full_cal_mean': ext_full_cal_mean, 'Ext_full_cal_std': ext_full_cal_std,
           'Ext_full_cal_eff': ext_full_cal_eff,
           'Bac_full_cal_mean': bac_full_cal_mean, 'Bac_full_cal_std': bac_full_cal_std,
           'Bac_full_cal_eff': bac_full_cal_eff,
           'Depol_full_cal_mean': depol_full_cal_mean, 'Depol_full_cal_std': depol_full_cal_std,
           'Depol_full_cal_eff': depol_full_cal_eff,
           'LR_full_cal_mean': lr_full_cal_mean, 'LR_full_cal_std': lr_full_cal_std, 'LR_full_cal_eff': lr_full_cal_eff
           }
    return retrieval, fit


def plot_rtv(retrieval, output_dir, name):
    """
    plot retrieval results
    :param retrieval: dict, retrieval results returned by the method inversion()
    :param output_dir: str, prefix of the figure name, indicating the saving directory
    :param name: str, name of the plot
    :return: filepath
    """
    if output_dir is None:
        filepath = name + '_retrieval.png'
    else:
        filepath = os.path.join(output_dir, name + '_retrieval.png')
        if retrieval.keys() == 2:
            print('There is nothing to plot because: ' + retrieval['rtv_message'])
        else:
            # about the table
            tab_rowlabel = ['mean', 'std', 'eff']
            tab_collabel = ['$V_t$', '$R_{eff}$', '$n$', '$\kappa$']
            tab_text = [
                ['%.2f' % retrieval['Vt_mean'], '%.2f' % retrieval['Reff_mean'], '%.2f' % retrieval['CRI_r_mean'],
                 '%.3f' % retrieval['CRI_i_mean']],
                ['%.3f' % retrieval['Vt_std'], '%.3f' % retrieval['Reff_std'], '%.3f' % retrieval['CRI_r_std'],
                 '%.4f' % retrieval['CRI_i_std']],
                ['%.2f' % retrieval['Vt_eff'], '%.2f' % retrieval['Reff_eff'], ' ', ' ']]
            fontsize = 14
            labelsize = 12
            font_label = FontProperties(weight='semibold', size=fontsize, variant='small-caps')
            # process_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            '''now = datetime.datetime.now()
            prs_time = now.strftime('%Y-%m-%dT%H:%M:%S')
            local_now = now.astimezone()
            tz_str = local_now.tzinfo.tzname(local_now)
            plt.figtext(0.8, 0.01, "Processed at : %s (%s time)" % (prs_time, tz_str), ha="center", fontsize=10,
                        color='gray')'''

            fig1 = plt.figure(figsize=(9, 5))  # , constrained_layout=True)
            gs = GridSpec(5, 5, figure=fig1)
            ax_left = fig1.add_subplot(gs[:, :3])
            ax_left.grid()
            ax_upright = fig1.add_subplot(gs[:2, 3:])
            ax_upright.grid()
            ax_upright.grid(which='minor', ls='--')
            ax_midright = fig1.add_subplot(gs[2: 4, 3:])
            ax_lowrigh = fig1.add_subplot(gs[-1, 3:])

            ax_left.errorbar(retrieval['r_grid'], retrieval['VSD_mean'], yerr=retrieval['VSD_std'], marker='o',
                             capsize=3)
            ax_left.tick_params(labelsize=labelsize)
            ax_left.set_xlabel('r $({\mu}m)$', fontsize=fontsize)  # fontproperties=font_label)
            ax_left.set_ylabel('dV/dlnr $({\mu}m^3/cm^3)$', fontsize=fontsize)  # fontproperties=font_label)
            ax_left.set_xscale('log')

            ax_upright.errorbar(retrieval['Wave_union'], retrieval['SSA_full_mean'], yerr=retrieval['SSA_full_std'],
                                marker='o',
                                capsize=3)
            ax_upright.tick_params(labelsize=labelsize)
            ax_upright.set_xlabel('${\lambda}({\mu}m)$', fontsize=fontsize)  # fontproperties=font_label)
            ax_upright.set_ylabel('SSA', fontsize=fontsize)  # fontproperties=font_label)
            ax_upright.yaxis.set_minor_locator(AutoMinorLocator(2))

            ax_tab = ax_midright.table(cellText=tab_text, cellLoc='center', rowLabels=tab_rowlabel,
                                       colLabels=tab_collabel, rowLoc='center', colLoc='center', loc='center',
                                       fontsize=fontsize)
            ax_tab.scale(1, 2)
            ax_midright.axis('off')

            ax_lowrigh.text(0, 0.75, 'Number of IS: %d\n$V_t$ in ${\mu}m^3/cm^3$, $R_{eff}$ in ${\mu}m$' % (
                retrieval['NIS']), fontsize=labelsize, transform=ax_lowrigh.transAxes)
            font_strtime = FontProperties(style='italic')
            ax_lowrigh.text(0, 0, retrieval['str_time'], fontsize=labelsize - 2, transform=ax_lowrigh.transAxes,
                            color='gray', fontproperties=font_strtime)
            ax_lowrigh.set_frame_on(False)
            ax_lowrigh.xaxis.set_visible(False)
            ax_lowrigh.yaxis.set_visible(False)
            fig1.suptitle(name + '_retrieval', fontproperties=font_label)
            plt.subplots_adjust(wspace=1.1, hspace=0.4)
            plt.savefig(filepath, dpi=300)
            plt.close()
    return filepath


def plot_fit(fit, output_dir, name):
    """
    plot fitting result
    :param fit: dict, fitted and recalculated optical properties returned by the method inversion()
    :param output_dir: str, prefix of the figure name, indicating the saving directory
    :param name: str, name of the plot
    :return: filepath
    """
    if output_dir is None:
        filepath = name + '_fit.png'
    else:
        filepath = os.path.join(output_dir, name + '_fit.png')
        if fit.keys() == 2:
            print('There is nothing to plot because: ' + fit['rtv_message'])
        else:
            fontsize = 14
            labelsize = 12
            markersize = 8
            font_label = FontProperties(weight='semibold', size=14, variant='small-caps')
            # process_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            ext_wave = fit['Ext_meas'].index
            bac_wave = fit['Bac_meas'].index
            lr_wave = fit['LR_meas'].index

            fig2, ax2 = plt.subplots(2, 2, figsize=(9, 5))  # , constrained_layout=True)
            for axes in ax2.ravel():
                axes.tick_params(labelsize=labelsize)
                axes.set_xlabel('Wavelength $({\mu}m)$', fontsize=fontsize)  # fontproperties=font_label)
                axes.xaxis.set_minor_locator(AutoMinorLocator(2))
                axes.yaxis.set_minor_locator(AutoMinorLocator(2))
                axes.grid()
                axes.grid(which='minor', ls='--')
            ax2[0, 0].errorbar(ext_wave, fit['Ext_meas'].values, yerr=fit['Ext_meas_err'] * fit['Ext_meas'].values,
                               marker='o', color='C0',
                               markersize=markersize, linestyle='', label='Measured', capsize=3)
            ax2[0, 0].errorbar(fit['Wave_union'], fit['Ext_full_cal_mean'], yerr=fit['Ext_full_cal_std'], marker='o',
                               markerfacecolor='none', markersize=markersize, linestyle='', label='Cal. mean',
                               capsize=3, color='C3')
            # ax2[0, 0].plot(fit['Wave_union'], fit['Ext_full_cal_eff'], marker='D', markerfacecolor='none', linestyle='',
            # markersize=markersize, label='Cal. effective')
            ax2[0, 0].set_ylabel(r'${\alpha}(Mm^{-1})$', fontsize=fontsize)  # fontproperties=font_label)
            ax2[0, 0].legend(loc='best', title='Residual: {:.2f}%'.format(fit['Residual_mean']))

            ax2[0, 1].errorbar(bac_wave, fit['Bac_meas'].values, yerr=fit['Bac_meas_err'] * fit['Bac_meas'].values,
                               marker='o', color='C0',
                               markersize=markersize, linestyle='', capsize=3)
            ax2[0, 1].errorbar(fit['Wave_union'], fit['Bac_full_cal_mean'], yerr=fit['Bac_full_cal_std'], marker='o',
                               markerfacecolor='none', markersize=markersize, linestyle='', capsize=3, color='C3')
            # ax2[0, 1].plot(fit['Wave_union'], fit['Bac_full_cal_eff'], marker='D', markerfacecolor='none',
            # markersize=markersize, linestyle='')
            ax2[0, 1].set_ylabel(r'${\beta}((Mm*sr)^{-1})$', fontsize=fontsize)  # fontproperties=font_label)

            ax2[1, 0].errorbar(lr_wave, fit['LR_meas'].values, yerr=fit['LR_err'] * fit['LR_meas'].values, marker='o',
                               linestyle='', color='C0',
                               markersize=markersize, capsize=3)
            ax2[1, 0].errorbar(fit['Wave_union'], fit['LR_full_cal_mean'], yerr=fit['LR_full_cal_std'], marker='o',
                               markerfacecolor='none', markersize=markersize, linestyle='', capsize=3, color='C3')
            # ax2[1, 0].plot(fit['Wave_union'], fit['LR_full_cal_eff'], marker='D', markerfacecolor='none', markersize=markersize, linestyle='')
            ax2[1, 0].set_ylabel('Lidar ratio (sr)', fontsize=fontsize)  # fontproperties=font_label)

            if len(fit['Depol_meas']):
                depol_wave = fit['Depol_meas'].index
                ax2[1, 1].errorbar(depol_wave, fit['Depol_meas'],
                                   yerr=fit['Depol_err'] * fit['Depol_meas'], marker='o',
                                   markersize=markersize, linestyle='', capsize=3, color='C0')
            ax2[1, 1].errorbar(fit['Wave_union'], fit['Depol_full_cal_mean'], yerr=fit['Depol_full_cal_std'],
                               marker='o', color='C3', markersize=markersize, markerfacecolor='none', linestyle='', capsize=3)
            ax2[1, 1].set_ylabel('${\delta}$', fontsize=fontsize)  # fontproperties=font_label)
            fig2.suptitle(name + '_fit', fontproperties=font_label)
            plt.subplots_adjust(wspace=0.3, hspace=0)
            plt.savefig(filepath, dpi=300)
            plt.close()
    return filepath


def txt_header():
    return f"""BOREAL

Data policy: The services and data that you have accessed are provided by the Laboratoire d'Optique Atmospherique, 
a joint research unit of the University of Lille and CNRS. If you utilize the aerosol products for publication purposes, 
we kindly request you to cite the paper by Chang et al., 2022 and acknowledge the contribution of 
"University of Lille/CNRS/Laboratoire d'Optique Atmospherique for BOREAL online processing service." 
Additionally, we encourage you to consider offering co-authorship to the scientists who contributed to the development 
of BOREAL, if their involvement is relevant to your work. Your recognition and collaboration contribute to the 
advancement of scientific research and the acknowledgment of the efforts invested in the development of these 
resources for the community.

"""


def export_txt(retrieval, fit, product_output_dir, filename, header=False, site_info=None, timestamp_str=None):
    """
    export retrieval and fit to two txt files
    :param timestamp_str: str, processing time
    :param site_info: str, information on the measurement site
    :param filename: str, filename
    :param product_output_dir: str, output directory
    :param retrieval: retrieval results returned by the method inversion()
    :param fit: fitted and recalculated optical properties returned by the method inversion()
    :return: two txt files
    """
    if product_output_dir is not None:
        filepath = os.path.join(product_output_dir, filename + '_output.txt')

        f = open(filepath, 'w+')
        if header:
            f.write(txt_header())

        if site_info is not None:
            for ele in site_info:
                f.write("%s = %s\n" % (ele, str(site_info[ele])))

        if timestamp_str is not None:
            timestamp_str = timestamp_str.replace(',', '_')
            timestamp_str = timestamp_str.replace(':', '-')
            f.write("Average Time = %s\n" % timestamp_str)

        f.write('\n' + "*" * 30 + "" + "   Retrievals   " + "*" * 30 + '\n')
        if fit is None:
            f.write(retrieval['rtv_message'] + '\n')
            f.write(retrieval['str_time'])
        else:
            f.write(retrieval['fwd_model'] + ' model is used.\n')
            f.write('Num. of selected individual solutions: {}\n'.format(retrieval['NIS']))
            f.write(
                'The following mean and std are statistical results of the individual solutions, not retrieval uncertainty\n')
            f.write('\tmean\tstd\n')
            f.write('mR\t{:.2f}\t{:.3f}\n'.format(retrieval['CRI_r_mean'], retrieval['CRI_r_std']))
            f.write('mI\t{:.3f}\t{:.4f}\n'.format(retrieval['CRI_i_mean'], retrieval['CRI_i_std']))
            f.write('Vt(um^3/cm^3)\t{:.2f}\t{:.3f}\n'.format(retrieval['Vt_mean'], retrieval['Vt_std']))
            f.write('Reff(um)\t{:.2f}\t{:.3f}\n'.format(retrieval['Reff_mean'], retrieval['Reff_std']))
            f.write('r(um)\tdv/dlnr_mean(um^3/cm^3)\tdv/dlnr_std(um^3/cm^3)\n')
            for r, vsd, vsd_std in zip(retrieval['r_grid'], retrieval['VSD_mean'], retrieval['VSD_std']):
                f.write('{:.5e}\t{:.5e}\t{:.5e}\n'.format(r, vsd, vsd_std))

            f.write('\n' + "*" * 30 + "" + "   Fittings   " + "*" * 30 + '\n')

            wave_list = [str(int(wave * 1000)) for wave in fit['Wave_union']]
            wave_str = 'wave(nm)\t'
            for wave in wave_list:
                wave_str += wave + '\t'
            f.write(wave_str + '\n')
            Meas_list = []
            Cal_list = []
            Cal_std_list = []
            Opt_type = ['Ext', 'Bac', 'LR']
            Opt_unit = ['(1/Mm)', '(1/(Mm*sr))', '(sr)']
            for j, opt in enumerate(Opt_type):
                meas = opt + '_meas' + Opt_unit[j] + '\t'
                cal = opt + '_cal\t'
                cal_std = opt + '_cal_std\t'
                meas_reindex = fit[opt + '_meas'].reindex(fit['Wave_union'], fill_value=-999)
                for i in range(len(wave_list)):
                    meas += '%.5e\t' % meas_reindex.iloc[i]
                    cal += '%.5e\t' % fit[opt + '_full_cal_mean'][i]
                    cal_std += '%.5e\t' % fit[opt + '_full_cal_std'][i]
                Meas_list.append(meas + '\n')
                Cal_list.append(cal + '\n')
                Cal_std_list.append(cal_std + '\n')
            for i in range(len(Opt_type)):
                f.write(Meas_list[i])
                f.write(Cal_list[i])
                f.write(Cal_std_list[i])
            meas = 'Depol_meas\t'
            cal = 'Depol_cal\t'
            cal_std = 'Depol_cal_std\t'
            if len(fit['Depol_meas']):
                meas_depol = fit['Depol_meas'].reindex(fit['Wave_union'], fill_value=-999).values
            else:
                meas_depol = [-999] * len(wave_list)
            for i in range(len(wave_list)):
                meas += '%.5e\t' % meas_depol[i]
                cal += '%.5e\t' % fit['Depol_full_cal_mean'][i]
                cal_std += '%.5e\t' % fit['Depol_full_cal_std'][i]
            f.write(meas + '\n')
            f.write(cal + '\n')
            f.write(cal_std + '\n')
            ssa_cal = 'SSA_cal\t'
            ssa_cal_std = 'SSA_cal_std\t'
            for i in range(len(wave_list)):
                ssa_cal += '%.5e\t' % retrieval['SSA_full_mean'][i]
                ssa_cal_std += '%.5e\t' % retrieval['SSA_full_std'][i]
            f.write(ssa_cal + '\n')
            f.write(ssa_cal_std + '\n')
            f.write(
                'Tot_residual\t{:.2f}\tTot_residual_std\t{:.3f}\n'.format(fit['Residual_mean'], fit['Residual_std']))

            f.write('\n' + "*" * 30 + "" + "   Message   " + "*" * 30 + '\n')
            f.write(retrieval['rtv_message'] + '\n')
            f.write(retrieval['str_time'])
    else:
        filepath = filename + '_output.txt'
    return filepath


# default_config, default_config2 = get_default_confg()
custom_config = {'rmin_min': 0.05, 'rmin_max': 0.3, 'rmax_min': 1.5, 'rmax_max': 15, 'r_regrid_num': 8, 'na': 1.5,
                 'sigma_na': 0.1, 'ka': 0.01, 'sigma_ka': 0.01, 'epsilon2': 2.5, 'epsilon_delta_x': 2.54,
                 'epsilon_delta_n': 0.07, 'epsilon_delta_k': 2.3}
