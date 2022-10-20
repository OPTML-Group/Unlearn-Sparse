from .GA import GA
from .RL import RL
from .fisher import fisher
from .retrain import retrain

def get_unlearn_method(name):
    """ method usage: (TODO)
    
    function(data_loaders, model, criterion, args)"""
    if name == "RL":
        return RL
    elif name == "GA":
        return GA
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")