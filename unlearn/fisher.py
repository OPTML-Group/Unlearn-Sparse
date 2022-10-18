import torch
from torch.autograd import grad
import pickle
import numpy as np
from time import time
from tqdm import tqdm







def fisher_information_martix(model,train_dl,device):
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter).to(device))
    total = 0
    for i,(data,label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        real_batch = data.shape[0]

        epsilon = 1e-7
        model.train()
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(prediction, model.parameters(), retain_graph=True, create_graph=False)
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon)**2
        total += real_batch
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i]/total

    return fisher_approximation



def fisher(forget_loader, retain_loader, test_loader, val_loader, model, criterion, optimizer, scheduler, args):
    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    fisher_approximation = fisher_information_martix(model,retain_loader,device)
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(args.alpha / fisher_approximation[i]).clamp(max=1e-3)*torch.empty_like(parameter).normal_(0,1)
        noise = noise * 10 if parameter.shape[-1] == 10 else noise

        parameter.data = parameter.data + noise
    return model






def gradient_norm(model):
    """Compute norm of gradient vector w.r.t. the model parameters."""
    gradient = torch.concat([p.grad.data.flatten() for p in model.parameters()])
    norm = torch.linalg.norm(gradient).tolist()
    return norm
