import time

import numpy as np
import torch
from tqdm import tqdm

# from sharpness_tools.math_utils import tridiag_to_eigv, hvp
# from sharpness_tools.utils import get_param_dim, get_device


def hvp(model, data_loader, vec):
    """
    Returns H*vec where H is the hessian of the loss w.r.t.
    the vectorized model parameters
    """
    hessian_vec_prod = None
    device = list(model.parameters())[0].device
    criterion = torch.nn.CrossEntropyLoss()

    for inputs, targets in data_loader:
        model.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        grad_dict = torch.autograd.grad(
            loss, model.parameters(), create_graph=True
        )
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        grad_grad = torch.autograd.grad(
            grad_vec, model.parameters(), grad_outputs=vec, only_inputs=True
        )
        if hessian_vec_prod is not None:
            hessian_vec_prod += torch.cat([g.contiguous().view(-1)
                                          for g in grad_grad])
        else:
            hessian_vec_prod = torch.cat(
                [g.contiguous().view(-1) for g in grad_grad])

    return hessian_vec_prod / len(data_loader)


def lanczos(model, data_loader, max_itr):
    """
    Lanczos iteration following the wikipedia article here
            https://en.wikipedia.org/wiki/Lanczos_algorithm
    :param model:
    :param data_loader:
    :param max_itr: max iteration
    :return: eigen values, weights
    """

    float_dtype = torch.float64

    model_dim = sum([p.numel() for p in model.parameters()])
    device = next(model.parameters()).device

    # Initializing empty arrays for storing
    tridiag = torch.zeros((max_itr, max_itr), dtype=float_dtype).to(device)
    vecs = torch.zeros((max_itr, model_dim), dtype=float_dtype).to(device)

    # intialize a random unit norm vector
    init_vec = torch.zeros(model_dim, dtype=float_dtype).uniform_(-1, 1)
    init_vec /= torch.norm(init_vec)
    vecs[0] = init_vec

    # placeholders for data
    beta = 0.0
    v_old = torch.zeros(model_dim, dtype=float_dtype).to(device)

    for k in range(max_itr):
        t = time.time()

        v = vecs[k]
        time_mvp = time.time()
        w = hvp(model, data_loader, v)
        w = w.type(float_dtype)
        time_mvp = time.time() - time_mvp

        w -= beta * v_old
        alpha = torch.dot(w, v)
        tridiag[k, k] = alpha
        w -= alpha * v

        # Reorthogonalization
        for j in range(k):
            tau = vecs[j]
            coeff = torch.dot(w, tau)
            w -= coeff * tau

        beta = torch.norm(w)

        if beta < 1e-6:
            raise ZeroDivisionError
            quit()

        if k + 1 < max_itr:
            tridiag[k, k + 1] = beta
            tridiag[k + 1, k] = beta
            vecs[k + 1] = w / beta

        v_old = v

        info = f"Iteration {k} / {max_itr} done in {time.time() - t:.2f}s (MVP: {time_mvp:.2f}s)"
        print(info)

    return vecs, tridiag
