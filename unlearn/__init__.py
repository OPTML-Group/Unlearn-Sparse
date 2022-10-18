import GA
import RL

def get_unlearn_method_iter(name):
    if name == "RL":
        return RL.RL_iter
    elif name == "GA":
        return GA.GA_iter
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")