import logging

from ..utils.collections import dotdict
from .builder.base.network import Network, Node
from .model import Model


class MultiModel(Model):
    """
    MultiModel base for heterogeneous networks.
    """

    @classmethod
    def from_node(cls, node):
        """
        Init MultiModel from Node.
        """
        assert isinstance(node, Node)
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return cls(node)

    def __init__(self, model_instance):
        assert isinstance(model_instance, (Node, Network))
        assert model_instance.initialised
        self.model_instance = model_instance

        # set model attributes
        self.name = self.model_instance.name
        self.state_vars = list(set.union(*[set(state_vars) for state_vars in self.model_instance.state_variable_names]))
        self.default_output = self.model_instance.default_output
        assert isinstance(self.default_output, str), "`default_output` must be a string."

        # create parameters
        self.params = self._create_hierarchical_params()

        # TODO resolve how to integrate in neurolib's fashion
        self.integration = None

        # create output and state dictionary
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()

        self.boldInitialized = False

        logging.info(f"{self.name}: Model initialized.")

    def _create_hierarchical_params(self):
        pass

    def getMaxDelay(self):
        return self.model_instance.max_delay
