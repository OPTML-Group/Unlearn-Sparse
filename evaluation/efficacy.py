import torch
from torch.autograd import grad


def gradient_norm(model):
    """Compute norm of gradient vector w.r.t. the model parameters."""
    gradient = torch.concat([p.grad.data.flatten()
                            for p in model.parameters()])
    norm = torch.linalg.norm(gradient).tolist()
    return norm


def efficacy_upper_bound(model, x, y, logistic_regression=False):
    """Return upper bound for forgetting score (efficacy)."""
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    predictions = model(x)
    if logistic_regression:
        predictions = torch.concat([-predictions, predictions], dim=-1)
    loss = ce_loss(predictions, y)
    model.zero_grad()
    loss.backward()
    squared_norm = gradient_norm(model) ** 2
    return torch.inf if squared_norm == 0 else 1. / squared_norm


def information_score(model, forget_loader, device, training=False):
    """
    Compute Fisher-based information score for the given model and data.
    The training argument determines if the resulting tensor requires grad and also if the computational graph should be created for the gradient.
    """
    # get model prediction)
    total = 0
    information = torch.tensor([0.], requires_grad=training).to(device)
    for i, (data, label) in enumerate(forget_loader):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        real_batch = data.shape[0]
        total += real_batch
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(prediction, model.parameters(),
                            retain_graph=True, create_graph=training)
            for derivative in gradient:
                information = information + torch.sum(derivative**2)
    information = information / total
    # return averaged information score
    return information


def efficacy(model, forget_loader, device):
    """Return forgetting score (efficacy)."""
    information_target_data = information_score(model, forget_loader, device)
    eff = torch.inf if information_target_data == 0 else 1. / information_target_data
    return eff.item()
