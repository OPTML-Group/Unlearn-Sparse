import torch
import numpy as np
from sklearn.svm import SVC
import torch.nn.functional as F

def entropy(p, dim = -1, keepdim = False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    prob = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch

            with torch.no_grad():
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def SVC_MIA(retain_loader_train, retain_loader_test, forget_loader, test_loader, model, device):
    shadow_train_prob = collect_prob(retain_loader_train, model)
    shadow_test_prob = collect_prob(test_loader, model)

    target_train_prob = collect_prob(retain_loader_test, model)
    target_test_prob = collect_prob(forget_loader, model)

    X_shadow = torch.cat([entropy(shadow_train_prob), entropy(shadow_test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_shadow = np.concatenate([np.ones(len(shadow_train_prob)), np.zeros(len(shadow_test_prob))])

    X_target_train = entropy(target_train_prob).cpu().numpy().reshape(-1, 1)
    X_target_test = entropy(target_test_prob).cpu().numpy().reshape(-1, 1)

    clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf.fit(X_shadow, Y_shadow)
    
    acc_train = clf.predict(X_target_train).mean()
    acc_test = 1 - clf.predict(X_target_test).mean()
    print("train acc: {}, test acc: {}, mean: {}".format(acc_train, acc_test, (acc_train + acc_test) / 2))
    return (acc_train, acc_test)