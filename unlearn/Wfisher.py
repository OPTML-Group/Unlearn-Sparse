import torch
from torch.autograd import grad
from tqdm import tqdm

def sam_grad(model,loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss,params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)

def apply_perturb(model,v):
    curr = 0 
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr:curr+length].data
        curr += length



        
def woodfisher(model,train_dl,device,criterion,v):
    model.eval()
    k_vec = torch.clone(v)
    N = len(train_dl)
    
    for idx,(data,label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output,label)
        sample_grad = sam_grad(model,loss)  
        if idx == 0:
            o_vec = torch.clone(sample_grad)
        else:
            tmp = torch.dot(o_vec, sample_grad)
            k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
            o_vec -= (tmp / (N + tmp)) * o_vec
    return k_vec


def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr:curr+length].view(param.shape)
            curr += length


def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = len(train_dl)

    for idx, (data, label) in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if idx == 0:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
    return k_vec


def Wfisher(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=128, shuffle=False)
    retain_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False)
    forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=128, shuffle=False)
    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()
    for i, (data, label) in enumerate(tqdm(forget_loader)):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss)*real_num
        forget_grad += f_grad
        total += real_num
    
    total_2 = 0
    for i, (data, label) in enumerate(tqdm(retain_grad_loader)):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss)*real_num
        retain_grad += r_grad
        total_2 += real_num
    retain_grad *= (total/((total+total_2)*total_2))
    forget_grad /= total+total_2
    perturb = woodfisher(model, retain_loader, device=device,
                         criterion=criterion, v=forget_grad-retain_grad)

    apply_perturb(model, args.alpha*perturb)

    return model
