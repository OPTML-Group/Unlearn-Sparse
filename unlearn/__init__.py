import GA
import RL

def get_unlearn_method(name):
    # method usage: (TODO)
    #   function(forget_loader, retain_loader, test_loader, val_loader, model, criterion, optimizer, scheduler, args)
    if name == "RL":
        return RL.RL
    elif name == "GA":
        return GA.GA
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")