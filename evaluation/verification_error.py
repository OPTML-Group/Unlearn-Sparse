import torch
def verification_error(model_retrain,model_unlearn):
    res = 0 
    for (para1,para2) in zip(model_retrain.parameters(),model_unlearn.parameters()):
        l2 = torch.linalg.norm(para1-para2)
        res += l2*l2
    return torch.sqrt(res).item()