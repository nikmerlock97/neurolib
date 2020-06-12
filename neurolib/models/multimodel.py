from .model import Model
from .builder.base.network import Node, Network


class MultiModel(Model):
    """
    MultiModel base for heterogeneous networks.
    """

    def __init__(self, model_instance, params=None):
        assert isinstance(model_instance, (Node, Network))
        self.model_instance = model_instance
        self.name = self.model_instance.name
        self.state_vars = self.model_instance.state_variable_names

    def getMaxDelay(self):
        return self.model_instance.max_delay
