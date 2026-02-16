#!.venv/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from boreal import BOREAL
from pandas import Series
from boreal.BOREAL import BorealException


class BorealTests(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #    cls.some_class_attribute = 0

    # @classmethod
    # def tearDownClass(cls):
    #    pass

    def setUp(self):
        expected_retrieval_keys = {'rtv_message', 'rtv_flag', 'str_time', 'fwd_model', 'NIS', 'r_grid', 'VSD_IS',
                                   'VSD_mean', 'VSD_std', 'CRI_r_mean', 'CRI_r_std', 'CRI_i_mean', 'CRI_i_std',
                                   'Vt_mean', 'Vt_std', 'Vt_eff', 'Reff_mean', 'Reff_std', 'Reff_eff', 'Wave_union',
                                   'SSA_full_mean', 'SSA_full_std', 'SSA_full_eff'}
        expected_fit_keys = {'rtv_message', 'str_time', 'Ext_meas', 'Ext_meas_err', 'Bac_meas', 'Bac_meas_err',
                             'Depol_meas', 'Depol_err', 'LR_meas', 'LR_err', 'Wave_union', 'Residual_mean',
                             'Residual_std', 'Residual_eff', 'Ext_full_cal_mean', 'Ext_full_cal_std',
                             'Ext_full_cal_eff', 'Bac_full_cal_mean', 'Bac_full_cal_std', 'Bac_full_cal_eff',
                             'Depol_full_cal_mean', 'Depol_full_cal_std', 'Depol_full_cal_eff', 'LR_full_cal_mean',
                             'LR_full_cal_std', 'LR_full_cal_eff'}
        ext_sim = {'355': 58.0121, '532': 61.4499, '1064': 18.4375}
        bac_sim = {'355': 1.1590, '532': 1.0714, '1064': 0.3688}
        depol_sim = {'355': 0.3132, '532': 0.3144, '1064': 0.2392}
        self.ext_sim = ext_sim
        self.bac_sim = bac_sim
        self.depol_sim = depol_sim
        self.expected_retrieval_keys = expected_retrieval_keys
        self.expected_fit_keys = expected_fit_keys

    def test_boreal_inversion_happy_path_spheroid_nonspectral_mI(self):
        # Arrange: inputs
        model = 'spheroid'
        aero_type = 'bba'  # 'bba', 'dust', 'urban', 'ss' (see salt)

        # Arrange: expected outputs
        expected_retrieval_keys = self.expected_retrieval_keys
        expected_fit_keys = self.expected_fit_keys

        # when dx=1e-5 in BOREAL.derivative
        # expected_vsd_mean = np.array([0., 0.02101806, 0.06109165, 0.10116153, 0.2993199, 0.44371504,
        # 2.08153182, 4.9535166, 5.64640924, 3.96482151, 3.89569719, 5.76772107,
        # 5.02646731, 2.05122003, 0.50428004, 0.23860125, 0.10188113, 0.06261273,
        # 0.04472683, 0.02976611, 0., 0., 0., 0.])

        expected_vsd_mean = np.array([0., 0.0210196, 0.0610862, 0.102949, 0.3005361, 0.4455101,
                                      2.080931, 4.9504778, 5.6456146, 3.9661874, 3.8994995, 5.7743655,
                                      5.0276256, 2.0463686, 0.5021457, 0.2393391, 0.1034472, 0.0650542,
                                      0.0456204, 0.0306294, 0., 0., 0., 0.])

        # Act
        actual_retrieval, actual_fit = BOREAL.inversion(ext=self.ext_sim, bac=self.bac_sim, depol=self.depol_sim,
                                                        aero_type=aero_type, model=model)

        # Assert
        self.assertTrue(type(actual_retrieval) == dict)
        self.assertTrue(type(actual_fit) == dict)

        actual_retrieval_keys = set(actual_retrieval.keys())
        actual_fit_keys = set(actual_fit.keys())
        self.assertEqual(expected_retrieval_keys, actual_retrieval_keys)
        self.assertEqual(expected_fit_keys, actual_fit_keys)

        actual_vsd_mean = actual_retrieval['VSD_mean']
        np.testing.assert_almost_equal(expected_vsd_mean, actual_vsd_mean)

    def test_boreal_inversion_happy_path_spheroid_spectral_mI(self):
        # Arrange: inputs
        model = 'spheroid'
        aero_type = 'dust'  # 'bba', 'dust', 'urban', 'ss' (see salt)

        # Arrange: expected outputs
        expected_retrieval_keys = self.expected_retrieval_keys
        expected_fit_keys = self.expected_fit_keys
        expected_vsd_mean = np.array([0., 0.0286248, 0.0319318, 0.0664842, 0.1375675, 0.3274778,
                                      2.7921674, 5.4438758, 4.722152, 3.4146416, 4.2937813, 5.7773991,
                                      4.2569871, 1.5842067, 0.4882438, 0.2517208, 0.1407428, 0.111725,
                                      0.0866893, 0.0693091, 0.0351905, 0., 0., 0.])

        # Act
        actual_retrieval, actual_fit = BOREAL.inversion(ext=self.ext_sim, bac=self.bac_sim, depol=self.depol_sim,
                                                        aero_type=aero_type, model=model)

        # Assert
        self.assertTrue(type(actual_retrieval) == dict)
        self.assertTrue(type(actual_fit) == dict)

        actual_retrieval_keys = set(actual_retrieval.keys())
        actual_fit_keys = set(actual_fit.keys())
        self.assertEqual(expected_retrieval_keys, actual_retrieval_keys)
        self.assertEqual(expected_fit_keys, actual_fit_keys)

        actual_vsd_mean = actual_retrieval['VSD_mean']
        np.testing.assert_almost_equal(expected_vsd_mean, actual_vsd_mean)

    def test_boreal_inversion_happy_path_sphere_nonspectral_mI(self):
        # Arrange: inputs
        model = 'sphere'
        aero_type = 'bba'  # 'bba', 'dust', 'urban', 'ss' (see salt)

        # Arrange: expected outputs
        expected_retrieval_keys = self.expected_retrieval_keys
        expected_fit_keys = self.expected_fit_keys
        expected_vsd_mean = np.array([0., 0.0517363, 0.152253, 0.4308567, 0.5619562, 1.1836335,
                                      4.0886363, 6.5427887, 4.7728518, 2.7185904, 3.590931, 3.9206347,
                                      1.9313191, 0.4393319, 0.2753303, 0.1464205, 0.1166809, 0.0843663,
                                      0.0299158, 0.035248, 0., 0., 0., 0.])

        # Act
        actual_retrieval, actual_fit = BOREAL.inversion(ext=self.ext_sim, bac=self.bac_sim, depol=self.depol_sim,
                                                        aero_type=aero_type, model=model)

        # Assert
        self.assertTrue(type(actual_retrieval) == dict)
        self.assertTrue(type(actual_fit) == dict)

        actual_retrieval_keys = set(actual_retrieval.keys())
        actual_fit_keys = set(actual_fit.keys())
        self.assertEqual(expected_retrieval_keys, actual_retrieval_keys)
        self.assertEqual(expected_fit_keys, actual_fit_keys)

        actual_vsd_mean = actual_retrieval['VSD_mean']
        np.testing.assert_almost_equal(expected_vsd_mean, actual_vsd_mean)

    def test_boreal_inversion_happy_path_sphere_spectral_mI(self):
        # Arrange: inputs
        model = 'sphere'
        aero_type = 'dust'  # 'bba', 'dust', 'urban', 'ss' (see salt)

        # Arrange: expected outputs
        expected_retrieval_keys = self.expected_retrieval_keys
        expected_fit_keys = self.expected_fit_keys
        expected_vsd_mean = np.array([0., 0.0234736, 0.1058703, 0.2877987, 0.5806688, 1.9010381,
                                      4.7010648, 7.1615518, 4.5972076, 1.905051, 0.5563875, 0.1731821,
                                      0.0652069, 0.0256304, 0.0206949, 0.0187218, 0.019818, 0.026435,
                                      0.0271352, 0.0381024, 0.0342559, 0.0241221, 0.0151959, 0.])

        # Act
        actual_retrieval, actual_fit = BOREAL.inversion(ext=self.ext_sim, bac=self.bac_sim, depol=self.depol_sim,
                                                        aero_type=aero_type, model=model)

        # Assert
        self.assertTrue(type(actual_retrieval) == dict)
        self.assertTrue(type(actual_fit) == dict)

        actual_retrieval_keys = set(actual_retrieval.keys())
        actual_fit_keys = set(actual_fit.keys())
        self.assertEqual(expected_retrieval_keys, actual_retrieval_keys)
        self.assertEqual(expected_fit_keys, actual_fit_keys)

        actual_vsd_mean = actual_retrieval['VSD_mean']
        np.testing.assert_almost_equal(expected_vsd_mean, actual_vsd_mean)

    def test_boreal_inversion_empty_extbac(self):
        model = 'sphere'
        aero_type = 'dust'  # 'bba', 'dust', 'urban', 'ss' (see salt)
        ext_empty = {}
        ext_all_negative = dict(-Series(self.ext_sim))

        kwargs1 = {'ext': ext_empty, 'bac': self.bac_sim, 'depol': self.depol_sim, 'aero_type': aero_type,
                   'model': model}
        kwargs2 = {'ext': ext_all_negative, 'bac': self.bac_sim, 'depol': self.depol_sim, 'aero_type': aero_type,
                   'model': model}

        expected_error_message1 = '.*measurements cannot be empty.$'
        #expected_error_message2 = '^All.*measurements are negative.$'

        self.assertRaisesRegex(BorealException, expected_error_message1, BOREAL.inversion, **kwargs1)
        #self.assertRaisesRegex(BorealException, expected_error_message2, BOREAL.inversion, **kwargs2)

    def test_boreal_inversion_negative_input(self):
        expected_error_message = '.*is negative.$'
        model = 'spheroid'
        aero_type = 'dust'
        Meas = [self.ext_sim, self.bac_sim, self.depol_sim]
        for meas in Meas:
            meas['355'] *= -1
            kwargs = {'ext': Meas[0], 'bac': Meas[1], 'depol': Meas[2], 'aero_type': aero_type, 'model': model}
            self.assertRaisesRegex(BorealException, expected_error_message, BOREAL.inversion, **kwargs)


if __name__ == '__main__':
    unittest.main()
