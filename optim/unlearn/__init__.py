from .fisher import fisher, fisher_new
from .FT import FT, FT_l1
from .FT_prune import FT_prune
from .FT_prune_bi import FT_prune_bi
from .GA import GA, GA_l1
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .retrain import retrain
from .retrain_ls import retrain_ls
from .retrain_sam import retrain_sam
from .Wfisher import Wfisher


def raw(data_loaders, model, criterion, args):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "FT_prune":
        return FT_prune
    elif name == "FT_prune_bi":
        return FT_prune_bi
    elif name == "GA_l1":
        return GA_l1
    elif name == "retrain_ls":
        return retrain_ls
    elif name == "retrain_sam":
        return retrain_sam
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
