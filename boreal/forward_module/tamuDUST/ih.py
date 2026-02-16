# this module contains functions for using the Irregular-Hexahedral scattering model (TAMUdust2020: Saito et al., 2021)
# use the sub-dataset extracted from TAMUdust-2020 for forward_module calculation

import os
import numpy as np
import math
from scipy.interpolate import interp1d


def __interp_nk(ker, eta, kappa):
    """
    implement 2-dim interpolation to obtain the kernel with specified (eta, kappa)
    :param kappa: float, imaginary part of the CRI
    :param eta: float, Real part of the CRI
    :param ker: kernel matrix to interpolate with shape (mR, mI, r)
    :return:
    """
    ii = max(mR[mR < eta].size - 1, 0)
    jj = max(mI[mI < kappa].size - 1, 0)
    n_left, n_right = mR[ii], mR[ii + 1]
    k_left, k_right = mI[jj], mI[jj + 1]
    func1 = interp1d([math.log(k_left), math.log(k_right)], np.array([ker[ii, jj], ker[ii, jj + 1]]), axis=0)  # linear interp for k on the log scal
    kernel_nleft_k = func1(math.log(kappa))
    func2 = interp1d([math.log(k_left), math.log(k_right)], np.array([ker[ii + 1, jj], ker[ii + 1, jj + 1]]), axis=0)
    kernel_nright_k = func2(math.log(kappa))  # linear interp for k on the log scale
    func3 = interp1d([n_left, n_right], np.array([kernel_nleft_k, kernel_nright_k]), axis=0)
    kernel_eta_kappa = func3(eta)
    return kernel_eta_kappa


'''def get_opt2(x_array, r_grid, only_fwd=False, wave_ext=None, wave_bac=None, wave_depol=None, lnr_grid_redu=None):
    """
        get bulk optical properties for a fixed sphericity.

        :param x_array: ndarray, x[:-2]: particle volume size distribution (VSD) with respect to volume-equivalent radius, r_ve: v(r) = dV/dlnr;
                              x[-2]: mR; x[-1]: mI
        :param r_grid: ndarray, r_ve grid. The first and last elements must coincide with the elements in r_grid_org.
        :param lnr_grid_redu: ndarray, logarithm of reduced radius grids for inversion, r_grid[0] and r_grid[-1] are included. These grids are logarithmic equidistant in [r_grid[0], r_grid[-1]].
                    the length should be equal to the length of dV_dlnr.
        :param wave_bac: list, wavelengths of backscattering
        :param wave_ext: list, wavelengths of extinction
        :param wave_depol: list, wavelength of depolarization
        :param only_fwd: whether it is only used for forward_module calculation. If True, wave + ext + bac + ssa + lr (+ pldr) at full waves will be output;
        otherwise, a ndarray of simulated measurements matching the input [ext, bac(, depol)] will be output for fitting.
        :return:
    """

    v_lnr_ = x_array[:-2]  # load VSD
    lnr_grid = np.log(r_grid)
    ind_n_left = mR[mR < x_array[-2]].size - 1
    ind_k_left = mI[mI < x_array[-1]].size - 1
    lnk = math.log(x_array[-1])
    lnk_left, lnk_right = math.log(mI[ind_k_left]), math.log(mI[ind_k_left + 1])
    n_left, n_right = mR[ind_n_left], mR[ind_n_left + 1]
    interp_fun1 = interp1d([lnk_left, lnk_right], np.array([OptProp_DB2[ind_n_left, ind_k_left], OptProp_DB2[ind_n_left, ind_k_left+1]]), axis=0)
    OptProp_nleft_k = interp_fun1(lnk)
    interp_fun2 = interp1d([lnk_left, lnk_right], np.array([OptProp_DB2[ind_n_left+1, ind_k_left], OptProp_DB2[ind_n_left+1, ind_k_left+1]]), axis=0)
    OptProp_nright_k = interp_fun2(lnk)
    interp_fun3 = interp1d([n_left, n_right], np.array([OptProp_nleft_k, OptProp_nright_k]), axis=0)
    Sub_OptProp_DB = interp_fun3(x_array[-2])

    Sub_OptProp_DB = Sub_OptProp_DB[:, (r_grid_org >= r_grid[0]) & (r_grid_org <= r_grid[-1]), :]

    if lnr_grid_redu is not None:
        v_lnr_interpd = np.interp(lnr_grid, lnr_grid_redu, v_lnr_)
        Sub_OptProp_DB_bulk = np.sum(Sub_OptProp_DB * v_lnr_interpd[:, np.newaxis], axis=1)
    else:
        Sub_OptProp_DB_bulk = np.sum(Sub_OptProp_DB * v_lnr_[:, np.newaxis], axis=1)

    ext = Sub_OptProp_DB_bulk[:, 0]
    bac = Sub_OptProp_DB_bulk[:, 1]
    sca = Sub_OptProp_DB_bulk[:, 2]
    ssa = sca / ext
    LidarRatio = ext / bac
    P11_pi = Sub_OptProp_DB_bulk[:, 3]
    P22_pi = Sub_OptProp_DB_bulk[:, 4]
    pldr = (P11_pi - P22_pi) / (P11_pi + P22_pi)

    if only_fwd:
        wave_full = [0.355, 0.532, 1.064]
        return wave_full, ext, bac, ssa, LidarRatio, pldr
    else:
        ext_ind = [Wave.index(w) for w in wave_ext]
        bac_ind = [Wave.index(w) for w in wave_bac]
        if wave_depol is None:
            opt = np.concatenate((ext[ext_ind], bac[bac_ind]))
        else:
            depol_ind = [Wave.index(w) for w in wave_depol]
            opt = np.concatenate((ext[ext_ind], bac[bac_ind], pldr[depol_ind]))
        return opt'''


def get_opt(x_array, r_grid, only_fwd=False, wave_ext=None, wave_bac=None, wave_depol=None, lnr_grid_redu=None, mI_dusttype=False, model='ih'):
    """
        get bulk optical properties for a fixed sphericity=0.695.
        :param model: reserved argument, in order to keep consistent with the input of spheroid.get_opt
        :param mI_dusttype: if consider mI spectral dependency of the dust type
        :param x_array: ndarray, x[:-2]: particle volume size distribution (VSD) with respect to volume-equivalent radius, r_ve: v(r) = dV/dlnr;
                              x[-2]: mR; x[-1]: mI
        :param r_grid: ndarray, r_ve grid. The first and last elements must coincide with the elements in r_grid_org.
        :param lnr_grid_redu: ndarray, logarithm of reduced radius grids for inversion, r_grid[0] and r_grid[-1] are included. These grids are logarithmic equidistant in [r_grid[0], r_grid[-1]].
                    the length should be equal to the length of dV_dlnr.
        :param wave_bac: list, wavelengths of backscattering
        :param wave_ext: list, wavelengths of extinction
        :param wave_depol: list, wavelength of depolarization
        :param only_fwd: whether it is only used for forward_module calculation. If True, wave + ext + bac + ssa + lr (+ pldr) at full waves will be output;
        otherwise, a ndarray of simulated measurements matching the input [ext, bac(, depol)] will be output for fitting.
        :return:
    """

    v_lnr_ = x_array[:-2]  # load VSD
    lnr_grid = np.log(r_grid)
    OptProp_DB2_bulk = []
    if mI_dusttype:  # dust-type mI
        kappa = np.array([x_array[-1], x_array[-1] * 0.52, 0.001])
    else:
        kappa = np.ones(len(Wave)) * x_array[-1]
    for i in range(len(Wave)):
        Sub_OptProp_DB = __interp_nk(OptProp_DB2[:, :, i, :, :], x_array[-2], kappa[i])
        Sub_OptProp_DB = Sub_OptProp_DB[(r_grid_org >= r_grid[0]) & (r_grid_org <= r_grid[-1])]
        if lnr_grid_redu is not None:
            v_lnr_interpd = np.interp(lnr_grid, lnr_grid_redu, v_lnr_)
            Sub_OptProp_DB_bulk = np.sum(Sub_OptProp_DB * v_lnr_interpd[:, np.newaxis], axis=0)
        else:
            Sub_OptProp_DB_bulk = np.sum(Sub_OptProp_DB * v_lnr_[:, np.newaxis], axis=0)
        OptProp_DB2_bulk.append(Sub_OptProp_DB_bulk)
    OptProp_DB2_bulk = np.array(OptProp_DB2_bulk)

    ext = OptProp_DB2_bulk[:, 0]
    bac = OptProp_DB2_bulk[:, 1]
    sca = OptProp_DB2_bulk[:, 2]
    ssa = sca / ext
    LidarRatio = ext / bac
    P11_pi = OptProp_DB2_bulk[:, 3]
    P22_pi = OptProp_DB2_bulk[:, 4]
    pldr = (P11_pi - P22_pi) / (P11_pi + P22_pi)

    if only_fwd:
        return Wave, ext, bac, ssa, LidarRatio, pldr
    else:
        ext_ind = [Wave.index(w) for w in wave_ext]
        bac_ind = [Wave.index(w) for w in wave_bac]
        if wave_depol is None:
            opt = np.concatenate((ext[ext_ind], bac[bac_ind]))
        else:
            depol_ind = [Wave.index(w) for w in wave_depol]
            opt = np.concatenate((ext[ext_ind], bac[bac_ind], pldr[depol_ind]))
        return opt


'''def get_optdust(x_array, r_grid, wave_ext, wave_bac, wave_depol=None, lnr_grid_redu=None, only_fwd=False):
    """
        get bulk optical properties for a fixed sphericity of 0.71.

        :param x_array: ndarray, x[:-2]: particle volume size distribution (VSD) with respect to volume-equivalent radius, r_ve: v(r) = dV/dlnr;
                              x[-2]: mR; x[-1]: mI
        :param r_grid: ndarray, r_ve grid. The first and last elements must coincide with the elements in r_grid_org.
        :param lnr_grid_redu: ndarray, logarithm of reduced radius grids for inversion, r_grid[0] and r_grid[-1] are included. These grids are logarithmic equidistant in [r_grid[0], r_grid[-1]].
                    the length should be equal to the length of dV_dlnr.
        :param wave_bac: list, wavelengths of backscattering
        :param wave_ext: list, wavelengths of extinction
        :param wave_depol: list, wavelength of depolarization
        :param only_fwd: whether it is only used for forward_module calculation. If True, wave + ext + bac + ssa + lr (+ pldr) at full waves will be output;
        otherwise, a ndarray of simulated measurements matching the input [ext, bac(, depol)] will be output for fitting.
        :return:
    """
    v_lnr_ = x_array[:-2]  # load VSD
    n = x_array[-2]
    lnr_grid = np.log(r_grid)
    OptProp_DB2_bulk = []
    k_list = [x_array[-1], x_array[-1] * 0.52, 0.001]  # the coefficient is derived from regression analysis
    for i in range(len(Wave)):
        k = k_list[i]
        Sub_OptProp_DB = __interp_nk(OptProp_DB2[:, :, i, :, :], n, k)
        Sub_OptProp_DB = Sub_OptProp_DB[(r_grid_org >= r_grid[0]) & (r_grid_org <= r_grid[-1])]
        if lnr_grid_redu is not None:
            v_lnr_interpd = np.interp(lnr_grid, lnr_grid_redu, v_lnr_)
            Sub_OptProp_DB_bulk = np.sum(Sub_OptProp_DB * v_lnr_interpd[:, np.newaxis], axis=0)
        else:
            Sub_OptProp_DB_bulk = np.sum(Sub_OptProp_DB * v_lnr_[:, np.newaxis], axis=0)
        OptProp_DB2_bulk.append(Sub_OptProp_DB_bulk)
    OptProp_DB2_bulk = np.array(OptProp_DB2_bulk)

    ext = OptProp_DB2_bulk[:, 0]
    bac = OptProp_DB2_bulk[:, 1]
    sca = OptProp_DB2_bulk[:, 2]
    ssa = sca / ext
    LidarRatio = ext / bac
    P11_pi = OptProp_DB2_bulk[:, 3]
    P22_pi = OptProp_DB2_bulk[:, 4]
    pldr = (P11_pi - P22_pi) / (P11_pi + P22_pi)

    if only_fwd:
        wave_full = [0.355, 0.532, 1.064]
        return wave_full, ext, bac, ssa, LidarRatio, pldr
    else:
        ext_ind = [Wave.index(w) for w in wave_ext]
        bac_ind = [Wave.index(w) for w in wave_bac]
        if wave_depol is None:
            opt = np.concatenate((ext[ext_ind], bac[bac_ind]))
        else:
            depol_ind = [Wave.index(w) for w in wave_depol]
            opt = np.concatenate((ext[ext_ind], bac[bac_ind], pldr[depol_ind]))
        return opt'''


# read sub-database
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data')
sub_DB = np.load(os.path.join(data_path, 'OptProp_new.npy'))  # columns: ['wave', 'D', 'V', 'A', 'Q_ext', 'SSA', 'g', 'P11', 'P22', 'r_ve']
mR = np.arange(1.38, 1.71, 0.02).round(2)
mI = np.array([1.000000e-04, 5.000000e-04, 8.189469e-04, 1.341348e-03,
               2.196985e-03, 3.598428e-03, 5.893843e-03, 9.653489e-03,
               1.581139e-02, 2.589737e-02, 4.241714e-02, 6.947477e-02,
               0.1]).round(4)
Wave = [0.355, 0.532, 1.064]
r_grid_org = sub_DB[: 41, -1].round(4)
d_grid_org = sub_DB[: 41, 1].round(4)

OptProp_DB2 = np.load(os.path.join(data_path, 'OptProp2_Phi695.npy'))  # [mR, mI, wl, r_grid, opt_props(K_ext, K_bac, K_sca, K_P11_pi, K_P22_pi)]
