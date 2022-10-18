import unlearn.GA as GA
import unlearn.RL as RL
import unlearn.fisher as fisher
def get_unlearn_method(name):
    # method usage: (TODO)
    #   function(forget_loader, retain_loader, test_loader, val_loader, model, criterion, optimizer, scheduler, args)
    if name == "RL":
        return RL.RL
    elif name == "GA":
        return GA.GA
    elif name == "fisher":
        return fisher.fisher
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")