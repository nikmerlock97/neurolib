"""
Set of tests for adaptive exponential integrate-and-fire mean-field model.
"""

import unittest

import numba
import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.builder.adex import (
    DEFAULT_PARAMS_EXC,
    DEFAULT_PARAMS_INH,
    EXC,
    AdExNetwork,
    AdExNetworkNode,
    ExcitatoryAdExMass,
    InhibitoryAdExMass,
    _get_interpolation_values,
    _table_lookup,
)
from neurolib.models.builder.model_input import ZeroInput

DURATION = 100.0
DT = 0.1
CORR_THRESHOLD = 0.9

# dictionary as backend name: format in which the noise is passed
BACKENDS_TO_TEST = {
    "jitcdde": lambda x: x.as_cubic_splines(),
    "numba": lambda x: x.as_array(),
}


class TestAdExCallbacks(unittest.TestCase):
    SIGMA_TEST = 3.2
    MU_TEST = 1.7
    INTERP_EXPECTED = (37, 117, 0.8000000000000185, 0.7875000000002501)
    FIRING_RATE_EXPECTED = 0.09444942503533124
    VOLTAGE_EXPECTED = -56.70455755705249
    TAU_EXPECTED = 0.4487499999999963

    @classmethod
    def setUpClass(cls):
        cls.mass = ExcitatoryAdExMass()

    def test_get_interpolation_values(self):
        self.assertTrue(callable(_get_interpolation_values))
        print(type(_get_interpolation_values))
        self.assertTrue(isinstance(_get_interpolation_values, numba.core.registry.CPUDispatcher))
        interp_result = _get_interpolation_values(
            self.SIGMA_TEST, self.MU_TEST, self.mass.sigma_range, self.mass.mu_range, self.mass.d_sigma, self.mass.d_mu,
        )
        self.assertTupleEqual(interp_result, self.INTERP_EXPECTED)

    def test_table_lookup(self):
        self.assertTrue(callable(_table_lookup))
        self.assertTrue(isinstance(_table_lookup, numba.core.registry.CPUDispatcher))
        firing_rate = _table_lookup(
            self.SIGMA_TEST,
            self.MU_TEST,
            self.mass.sigma_range,
            self.mass.mu_range,
            self.mass.d_sigma,
            self.mass.d_mu,
            self.mass.firing_rate_cascade,
        )
        self.assertEqual(firing_rate, self.FIRING_RATE_EXPECTED)

        voltage = _table_lookup(
            self.SIGMA_TEST,
            self.MU_TEST,
            self.mass.sigma_range,
            self.mass.mu_range,
            self.mass.d_sigma,
            self.mass.d_mu,
            self.mass.voltage_cascade,
        )
        self.assertEqual(voltage, self.VOLTAGE_EXPECTED)

        tau = _table_lookup(
            self.SIGMA_TEST,
            self.MU_TEST,
            self.mass.sigma_range,
            self.mass.mu_range,
            self.mass.d_sigma,
            self.mass.d_mu,
            self.mass.tau_cascade,
        )
        self.assertEqual(tau, self.TAU_EXPECTED)


class ALNMassTestCase(unittest.TestCase):
    def _run_node(self, node, duration, dt):
        coupling_variables = {k: 0.0 for k in node.required_couplings}
        noise = ZeroInput(duration, dt, independent_realisations=node.num_noise_variables).as_cubic_splines()
        system = jitcdde_input(
            node._derivatives(coupling_variables), input=noise, callback_functions=node._callbacks(),
        )
        system.constant_past(np.array(node.initial_state))
        system.adjust_diff()
        times = np.arange(dt, duration + dt, dt)
        return np.vstack([system.integrate(time) for time in times])


class TestAdExMass(ALNMassTestCase):
    def _create_exc_mass(self):
        exc = ExcitatoryAdExMass()
        exc.index = 0
        exc.idx_state_var = 0
        exc.init_mass()
        return exc

    def _create_inh_mass(self):
        inh = InhibitoryAdExMass()
        inh.index = 0
        inh.idx_state_var = 0
        inh.init_mass()
        return inh

    def test_init(self):
        adex_exc = self._create_exc_mass()
        adex_inh = self._create_inh_mass()
        self.assertTrue(isinstance(adex_exc, ExcitatoryAdExMass))
        self.assertTrue(isinstance(adex_inh, InhibitoryAdExMass))
        self.assertDictEqual(adex_exc.parameters, DEFAULT_PARAMS_EXC)
        self.assertDictEqual(adex_inh.parameters, DEFAULT_PARAMS_INH)
        # test cascade
        np.testing.assert_equal(adex_exc.mu_range, adex_inh.mu_range)
        np.testing.assert_equal(adex_exc.sigma_range, adex_inh.sigma_range)
        np.testing.assert_equal(adex_exc.firing_rate_cascade, adex_inh.firing_rate_cascade)
        np.testing.assert_equal(adex_exc.voltage_cascade, adex_inh.voltage_cascade)
        np.testing.assert_equal(adex_exc.tau_cascade, adex_inh.tau_cascade)
        for adex in [adex_exc, adex_inh]:
            # test cascade
            self.assertTrue(callable(getattr(adex, "firing_rate_lookup")))
            self.assertTrue(callable(getattr(adex, "voltage_lookup")))
            self.assertTrue(callable(getattr(adex, "tau_lookup")))
            # test callbacks
            self.assertEqual(len(adex._callbacks()), 3)
            self.assertTrue(all(len(callback) == 3 for callback in adex._callbacks()))
            # test numba callbacks
            self.assertEqual(len(adex._numba_callbacks()), 3)
            for numba_callbacks in adex._numba_callbacks():
                self.assertEqual(len(numba_callbacks), 2)
                self.assertTrue(isinstance(numba_callbacks[0], str))
                self.assertTrue(isinstance(numba_callbacks[1], numba.core.registry.CPUDispatcher))
            # test derivatives
            coupling_variables = {k: 0.0 for k in adex.required_couplings}
            self.assertEqual(
                len(adex._derivatives(coupling_variables)), adex.num_state_variables,
            )
            self.assertEqual(len(adex.initial_state), adex.num_state_variables)
            self.assertEqual(len(adex.noise_input_idx), adex.num_noise_variables)

    def test_run(self):
        adex_exc = self._create_exc_mass()
        adex_inh = self._create_inh_mass()
        for adex in [adex_exc, adex_inh]:
            result = self._run_node(adex, DURATION, DT)
            self.assertTrue(isinstance(result, np.ndarray))
            self.assertTupleEqual(result.shape, (int(DURATION / DT), adex.num_state_variables))


class TestAdExNetworkNode(unittest.TestCase):
    def _create_node(self):
        np.random.seed(42)
        node = AdExNetworkNode()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        adex = self._create_node()
        self.assertTrue(isinstance(adex, AdExNetworkNode))
        self.assertEqual(len(adex), 2)
        self.assertDictEqual(adex[0].parameters, DEFAULT_PARAMS_EXC)
        self.assertDictEqual(adex[1].parameters, DEFAULT_PARAMS_INH)
        self.assertTrue(hasattr(adex, "_rescale_connectivity"))
        self.assertEqual(len(adex._sync()), 4 * len(adex))
        self.assertEqual(len(adex.default_network_coupling), 2)
        np.testing.assert_equal(
            np.array(sum([adexm.initial_state for adexm in adex], [])), adex.initial_state,
        )

    def test_run(self):
        adex = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = adex.run(
                DURATION, DT, noise_func(ZeroInput(DURATION, DT, adex.num_noise_variables)), backend=backend, dt=DT,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), adex.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in adex.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in adex.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestAdExNetwork(unittest.TestCase):
    SC = np.random.rand(2, 2)
    DELAYS = np.zeros((2, 2))

    def test_init(self):
        adex = AdExNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(adex, AdExNetwork))
        self.assertEqual(len(adex), self.SC.shape[0])
        self.assertEqual(adex.initial_state.shape[0], adex.num_state_variables)
        self.assertEqual(adex.default_output, f"q_mean_{EXC}")

    def test_run(self):
        np.random.seed(42)
        adex = AdExNetwork(self.SC, self.DELAYS)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = adex.run(
                DURATION, DT, noise_func(ZeroInput(DURATION, DT, adex.num_noise_variables)), backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), adex.num_state_variables / adex.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), adex.num_nodes) for result_ in result))
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
