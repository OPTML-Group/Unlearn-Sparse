import torch
from torch.autograd import grad
from tqdm import tqdm

def fisher_information_martix(model, train_dl, device):
    model.eval()
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
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(prediction, model.parameters(), retain_graph=True, create_graph=False)
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon) ** 2
        total += real_batch
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i] / total

    return fisher_approximation

def fisher(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]

    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    fisher_approximation = fisher_information_martix(model,retain_loader,device)
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(args.alpha / fisher_approximation[i]).clamp(max=1e-3)*torch.empty_like(parameter).normal_(0,1)
        noise = noise * 10 if parameter.shape[-1] == 10 else noise

        parameter.data = parameter.data + noise
    return model
