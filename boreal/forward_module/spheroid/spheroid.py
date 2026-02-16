# this module contains functions for using the Spheroid scattering model (Dubovik et al. 2006)
import os
import numpy as np
import math
from scipy.interpolate import interp1d


class BOREAL_FwdModelException(Exception):
    pass


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
    left_kappa = math.log(k_left)   # log interpolate in kappa grid
    right_kappa = math.log(k_right)
    x_kappa = math.log(kappa)
    func1 = interp1d([left_kappa, right_kappa], np.array([ker[ii, jj], ker[ii, jj + 1]]), axis=0)
    kernel_nleft_k = func1(x_kappa)
    func2 = interp1d([left_kappa, right_kappa], np.array([ker[ii + 1, jj], ker[ii + 1, jj + 1]]), axis=0)
    kernel_nright_k = func2(x_kappa)  # linear interp for k on the log scale
    func3 = interp1d([n_left, n_right], np.array([kernel_nleft_k, kernel_nright_k]), axis=0)
    kernel_eta_kappa = func3(eta)
    return kernel_eta_kappa


def __get_Kernel_matrix(eta, kappa, r_grid, wave_list, kernel):
    K_matrix = []
    if isinstance(kappa, float):
        kernel_nk_ = __interp_nk(kernel, eta, kappa)
        for wl in wave_list:
            r_equiv = r_grid * 0.34 / wl
            ln_r_equiv = np.log(r_equiv)
            kernel_equi = 1000 * (0.34 / wl) * np.interp(ln_r_equiv, ln_r_grid_org, kernel_nk_)  # log interpolation in radius grid
            K_matrix.append(kernel_equi)
    else:
        for i in range(len(wave_list)):
            kernel_nk_ = __interp_nk(kernel, eta, kappa[i])
            r_equiv = r_grid * 0.34 / wave_list[i]
            ln_r_equiv = np.log(r_equiv)
            kernel_equi = 1000 * (0.34 / wave_list[i]) * np.interp(ln_r_equiv, ln_r_grid_org, kernel_nk_)  # log interpolation in radius grid
            K_matrix.append(kernel_equi)
    K_matrix = np.mat(np.array(K_matrix))

    return K_matrix


def get_opt(x_array, r_grid, only_fwd=False, wave_ext=None, wave_bac=None, wave_depol=None, model='sphere', lnr_grid_redu=None, mI_dusttype=False):
    """
    the forward model to generate optical properties from micro-physical properties
    :param mI_dusttype: if consider mI spectral dependency of the dust type
    :param wave_bac: ndarray, wavelengths of backscattering
    :param wave_ext: ndarray, wavelengths of extinction
    :param wave_depol: list, wavelength of depolarization
    :param model: str, 'sphere' or 'spheroid', use sphere kernels ('sphere') or spheroid kernels ('spheroid')
    :param only_fwd: whether it is only used for forward calculation. If True, wave + ext + bac + ssa + lr (+ pldr) will be output;
    otherwise, ext + bac will be output for inversion.
    :param r_grid: ndarray, full-resolved grids
    :param lnr_grid_redu: ndarray, logarithm of reduced radius grids for inversion, r_grid[0] and r_grid[-1] are included. These grids are logarithmic equidistant in [r_grid[0], r_grid[-1]].
                    the length should be equal to the length of dV_dlnr.
    :param x_array: list or ndarray, values of each size bin, with the refractive index (x[-2:])
    :return: ndarray, fit optical data (3 beta + 2 alpha)
    """
    vsd = x_array[:-2]
    ln_r_grid = np.log(r_grid)  # both the kernel function and dV_dlnr should be interpolated in the log-scale
    if lnr_grid_redu is not None:
        # Kernel_matrix = Kernel_matrix * np.mat(rep_max)
        vsd_mat = np.mat(np.interp(ln_r_grid, lnr_grid_redu, vsd)).T
    else:
        vsd_mat = np.mat(vsd).T
    wave_full = [0.355, 0.532, 1.064]
    if only_fwd:  # only perform forward calculation, this will output ext_coe, bac_coe, ssa, lr and pldr in the full spectral range
        Opt = []
        if model == 'sphere':
            Kernel_list = [AKernel, BKernel, Abs_Kernel]
        else:  # model == 'spheroid'
            Kernel_list = [AKernel_299, BKernel_299, Abs_Kernel_299, BKernel_p22_299]
        if mI_dusttype:  # dust-type mI
            kappa = np.array([x_array[-1], x_array[-1] * 0.52, 0.001])
        else:  # non-dust-type mI
            kappa = x_array[-1]
        for kernel in Kernel_list:
            kernel_mat = __get_Kernel_matrix(eta=x_array[-2], kappa=kappa, r_grid=r_grid, wave_list=wave_full, kernel=kernel)
            Opt.append(np.array(kernel_mat * vsd_mat).ravel())
        ext = Opt[0]
        bac = Opt[1]
        lr = ext / bac
        absorb = Opt[2]
        ssa = (ext - absorb) / ext
        if model == 'sphere':
            depol = np.zeros(len(wave_full))
        else:
            depol = (bac - Opt[3]) / (bac + Opt[3])
        return wave_full, ext, bac, ssa, lr, depol

    else:  # the output is followed by measurement fitting
        # first: ext and bac
        if model == 'sphere':
            ext_Kernel = AKernel
            bac_Kernel = BKernel
            depol_flag = 0
        else:  # motel == 'spheroid'
            ext_Kernel = AKernel_299
            bac_Kernel = BKernel_299
            depol_flag = 1

        kwargs = {'eta': x_array[-2], 'r_grid': r_grid}
        if mI_dusttype:  # dust-type mI
            kappa = np.array([x_array[-1], x_array[-1] * 0.52, 0.001])
            kappa_ext = kappa[[wave_full.index(i) for i in wave_ext]]
            kappa_bac = kappa[[wave_full.index(i) for i in wave_bac]]
        else:  # non-dust-type mI
            kappa_ext, kappa_bac = x_array[-1], x_array[-1]
        ext_and_bac = []
        for kappa, wave, kernel in zip([kappa_ext, kappa_bac], [wave_ext, wave_bac], [ext_Kernel, bac_Kernel]):
            kernel_mat_ext_or_bac = __get_Kernel_matrix(kappa=kappa, wave_list=wave, kernel=kernel, **kwargs)
            ext_and_bac.append(np.array(kernel_mat_ext_or_bac * vsd_mat).ravel())
        if depol_flag and (wave_depol is not None):
            bulk_p11_and_p11 = []
            if mI_dusttype:  # dust-type mI
                kappa = np.array([x_array[-1], x_array[-1] * 0.52, 0.001])
                kappa_depol = kappa[[wave_full.index(i) for i in wave_depol]]
            else:  # non-dust-type mI
                kappa_depol = x_array[-1]
            for kernel_for_depol in [BKernel_299, BKernel_p22_299]:
                kernel_mat_for_depol = __get_Kernel_matrix(kappa=kappa_depol, wave_list=wave_depol, kernel=kernel_for_depol, **kwargs)
                bulk_p11_and_p11.append(np.array(kernel_mat_for_depol * vsd_mat).ravel())
            depol = (bulk_p11_and_p11[0] - bulk_p11_and_p11[1]) / (bulk_p11_and_p11[0] + bulk_p11_and_p11[1])
            opt = np.concatenate((np.concatenate(ext_and_bac), depol))
        else:
            opt = np.concatenate(ext_and_bac)
        return opt


# load grid refractive index, extinction and back scattering kernels
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data')
mR = np.load(os.path.join(data_path, 'mR.npy'))
mI = np.load(os.path.join(data_path, 'mI.npy'))
AKernel = np.load(os.path.join(data_path, 'AKernel.npy'))
BKernel = np.load(os.path.join(data_path, 'BKernel.npy'))
Abs_Kernel = np.load(os.path.join(data_path, 'Abs_Kernel.npy'))
AKernel_299 = np.load(os.path.join(data_path, 'AKernel_299.npy'))
BKernel_299 = np.load(os.path.join(data_path, 'BKernel_299.npy'))
Abs_Kernel_299 = np.load(os.path.join(data_path, 'Abs_Kernel_299.npy'))
BKernel_p22_299 = np.load(os.path.join(data_path, 'BKernel_p22_299.npy'))
r_grid_org = np.load(os.path.join(data_path, 'r_grid.npy'))
# r_grid_eff = r_grid_org[11: -2]
ln_r_grid_org = np.log(r_grid_org)
