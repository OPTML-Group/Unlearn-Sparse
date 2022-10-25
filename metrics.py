import numpy as np
import math
import torch
from torch.autograd import grad
from sklearn.svm import SVC
from tqdm import tqdm
import torch.nn.functional as F


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):

    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False)
    prob = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device)
                     for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]
                    ).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model)
    clf = SVC(C=3, gamma='auto', kernel='rbf')
    #clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


def membership_attack(retain_loader, forget_loader, test_loader, model):
    prob = get_membership_attack_prob(
        retain_loader, forget_loader, test_loader, model)
    print("Attack prob: ", prob)
    return prob


def collect_performance(data_loader, model):

    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False)
    prob = []
    label = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device)
                     for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
            label.append(target)
    return torch.cat(prob).cpu().numpy(), torch.cat(label).cpu().numpy()


def MIA(retain_loader_train, retain_loader_test, forget_loader, test_loader, model):
    shadow_train_performance = collect_performance(retain_loader_train, model)
    shadow_test_performance = collect_performance(test_loader, model)
    target_train_performance = collect_performance(retain_loader_test, model)
    target_test_performance = collect_performance(forget_loader, model)

    BBB = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                               target_train_performance, target_test_performance, num_classes=10)
    BBB._mem_inf_benchmarks()


class black_box_benchmarks(object):

    def __init__(self, shadow_train_performance, shadow_test_performance,
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)
                          == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)
                          == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)
                          == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)
                          == self.t_te_labels).astype(int)

        self.s_tr_conf = np.array(
            [self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array(
            [self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array(
            [self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array(
            [self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(
            self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(
            self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(
            self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(
            self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(
            true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(
            true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values < value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(
            acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc))
        return

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(
                s_tr_values[self.s_tr_labels == num], s_te_values[self.s_te_labels == num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) +
                           t_te_non_mem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(
            n=v_name, acc=mem_inf_acc))
        return

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._mem_inf_thre('confidence', self.s_tr_conf,
                               self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -
                               self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -
                               self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return


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
