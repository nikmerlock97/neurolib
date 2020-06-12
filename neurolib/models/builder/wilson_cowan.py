"""
Wilson-Cowan model.

Main reference:
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory
    interactions in localized populations of model neurons. Biophysical journal,
    12(1), 1-24.

Additional reference:
    Papadopoulos, L., Lynn, C. W., Battaglia, D., & Bassett, D. S. (2020).
    Relations between large scale brain connectivity and effects of regional
    stimulation depend on collective dynamical state. arXiv preprint
    arXiv:2002.00094.
"""

import numpy as np
from jitcdde import input as system_input
from symengine import exp

from ..builder.base.network import Network, SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import EXC, INH, NeuralMass

DEFAULT_PARAMS_EXC = {"a": 1.5, "mu": 3.0, "tau": 2.5, "ext_input": 0.6}
DEFAULT_PARAMS_INH = {"a": 1.5, "mu": 3.0, "tau": 3.75, "ext_input": 0.0}
# matrix as [from, to], masses as (EXC, INH)
DEFAULT_WC_NODE_CONNECTIVITY = np.array([[16.0, 15.0], [12.0, 3.0]])


class WilsonCowanMass(NeuralMass):
    """
    Wilson-Cowan neural mass. Can be excitatory or inhibitory, depending on the
    parameters.
    """

    name = "Wilson-Cowan mass"
    label = "WCmass"

    num_state_variables = 1
    num_noise_variables = 1
    coupling_variables = {0: "q_mean"}
    state_variable_names = ["q_mean"]
    required_parameters = ["a", "mu", "tau", "ext_input"]

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [0.05 * np.random.uniform(0, 1)]

    def _sigmoid(self, x):
        return 1.0 / (1.0 + exp(-self.parameters["a"] * (x - self.parameters["mu"])))


class ExcitatoryWilsonCowanMass(WilsonCowanMass):
    """
    Excitatory Wilson-Cowan neural mass.
    """

    name = "Wilson-Cowan excitatory mass"
    label = f"WCmass{EXC}"
    coupling_variables = {0: f"q_mean_{EXC}"}
    state_variable_names = [f"q_mean_{EXC}"]
    mass_type = EXC
    required_couplings = ["node_exc_exc", "node_inh_exc", "network_exc_exc"]

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters or DEFAULT_PARAMS_EXC)

    def _derivatives(self, coupling_variables):
        [x] = self._unwrap_state_vector()
        d_x = (
            -x
            + (1.0 - x)
            * self._sigmoid(
                coupling_variables["node_exc_exc"]
                - coupling_variables["node_inh_exc"]
                + coupling_variables["network_exc_exc"]
                + self.parameters["ext_input"]
            )
            + system_input(self.noise_input_idx[0])
        ) / self.parameters["tau"]

        return [d_x]


class InhibitoryWilsonCowanMass(WilsonCowanMass):

    name = "Wilson-Cowan inhibitory mass"
    label = f"WCmass{INH}"
    coupling_variables = {0: f"q_mean_{INH}"}
    state_variable_names = [f"q_mean_{INH}"]
    mass_type = INH
    required_couplings = ["node_exc_inh", "node_inh_inh"]

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters or DEFAULT_PARAMS_INH)

    def _derivatives(self, coupling_variables):
        [x] = self._unwrap_state_vector()
        d_x = (
            -x
            + (1.0 - x)
            * self._sigmoid(
                coupling_variables["node_exc_inh"] - coupling_variables["node_inh_inh"] + self.parameters["ext_input"]
            )
            + system_input(self.noise_input_idx[0])
        ) / self.parameters["tau"]

        return [d_x]


class WilsonCowanNetworkNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Default Wilson-Cowan network node with 1 excitatory and 1 inhibitory
    population.
    """

    name = "Wilson-Cowan node"
    label = "WCnode"

    def __init__(
        self, exc_paramaters=None, inh_parameters=None, connectivity=DEFAULT_WC_NODE_CONNECTIVITY,
    ):
        """
        :param exc_parameters: parameters for the excitatory mass
        :type exc_parameters: dict|None
        :param inh_parameters: parameters for the inhibitory mass
        :type inh_parameters: dict|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        """
        excitatory_mass = ExcitatoryWilsonCowanMass(exc_paramaters)
        excitatory_mass.index = 0
        inhibitory_mass = InhibitoryWilsonCowanMass(inh_parameters)
        inhibitory_mass.index = 1
        super().__init__(
            neural_masses=[excitatory_mass, inhibitory_mass],
            local_connectivity=connectivity,
            # within W-C node there are no local delays
            local_delays=None,
        )


class WilsonCowanNetwork(Network):
    """
    Whole brain network of Wilson-Cowan excitatory and inhibitory nodes.
    """

    name = "Wilson-Cowan network"
    label = "WCnet"

    sync_variables = ["network_exc_exc"]

    def __init__(
        self,
        connectivity_matrix,
        delay_matrix,
        exc_mass_parameters=None,
        inh_mass_parameters=None,
        local_connectivity=DEFAULT_WC_NODE_CONNECTIVITY,
    ):
        """
        :param connectivity_matrix: connectivity matrix for between nodes
            coupling, typically DTI structural connectivity, matrix as [from,
            to]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, typically derived from
            length matrix, if None, delays are all zeros, in ms, matrix as
            [from, to]
        :type delay_matrix: np.ndarray|None
        :param exc_mass_parameters: parameters for each excitatory Wilson-Cowan
            neural mass, if None, will use default
        :type exc_mass_parameters: list[dict]|dict|None
        :param inh_mass_parameters: parameters for each inhibitory Wilson-Cowan
            neural mass, if None, will use default
        :type inh_mass_parameters: list[dict]|dict|None
        :param local_connectivity: local within-node connectivity matrix
        :type local_connectivity: list[np.ndarray]|np.ndarray
        """
        num_nodes = connectivity_matrix.shape[0]
        exc_mass_parameters = self._prepare_mass_parameters(exc_mass_parameters, num_nodes)
        inh_mass_parameters = self._prepare_mass_parameters(inh_mass_parameters, num_nodes)
        local_connectivity = self._prepare_mass_parameters(local_connectivity, num_nodes, native_type=np.ndarray)

        nodes = []
        for i, (exc_params, inh_params, local_conn) in enumerate(
            zip(exc_mass_parameters, inh_mass_parameters, local_connectivity)
        ):
            node = WilsonCowanNetworkNode(
                exc_paramaters=exc_params, inh_parameters=inh_params, connectivity=local_conn,
            )
            node.index = i
            node.idx_state_var = i * node.num_state_variables
            # set correct indices of noise input
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
            nodes.append(node)

        super().__init__(
            nodes=nodes, connectivity_matrix=connectivity_matrix, delay_matrix=delay_matrix,
        )
        # assert we have only one sync variable
        assert len(self.sync_variables) == 1

    def _sync(self):
        # excitatory population within the node is first, hence the
        # within_node_idx is 0
        return self._additive_coupling(within_node_idx=0, symbol=self.sync_variables[0]) + super()._sync()
