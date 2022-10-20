from .GA import GA
from .RL import RL
from .FT import FT
from .fisher import fisher
from .retrain import retrain

def get_unlearn_method(name):
    """ method usage:
    
    function(data_loaders, model, criterion, args)"""
    if name == "RL":
        return RL
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")