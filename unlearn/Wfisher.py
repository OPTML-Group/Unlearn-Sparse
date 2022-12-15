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

def Wfisher(data_loaders,model,criterion,args):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    retain_loader = torch.utils.data.DataLoader(retain_loader.dataset, batch_size=32, shuffle=False)
    forget_loader = torch.utils.data.DataLoader(forget_loader.dataset, batch_size=128, shuffle=False)
    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    for i, (data,label) in enumerate(tqdm(forget_loader)):
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output,label)
        f_grad = sam_grad(model,loss)*real_num
        forget_grad += f_grad
        total+=real_num
    forget_grad /= total

    perturb = woodfisher(model,retain_loader,device=device,criterion=criterion,v=forget_grad)
    
    apply_perturb(model, args.alpha * perturb)

    return model



def woodfisher_batch(model,inputs,labels,device,criterion,v):
    model.eval()
    k_new = v
    params = []
    for param in model.parameters():
        params.append(param)
    N = inputs.shape[0]
    for i in range(N):
        data = inputs[i]
        label = labels[i]
        output = model(data)
        loss = criterion(output,label)
        sample_grad = grad(loss,params)
        if i == 0:
            o_new = sample_grad
        else:
            o_old = o_new
            k_old = k_new
            tmp = torch.matmul(o_old,sample_grad)
            o_new = o_old - (tmp/(N+tmp))*o_old
            k_new = k_old - (tmp/(N+tmp))*k_old
    return k_new