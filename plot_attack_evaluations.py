import os
import torch
import shutil
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt


pruning_methods = ['omp']  # ["synflow", "omp"]
sparsities = "0 0.5 0.75 0.9 0.95 0.99".split(' ')
methods = "FT FT_prune FT_prune_bi".split(
    ' ')  # fisher_new FT RL raw retrain wfisher
nums = [4500]  # [450, 2250, 4500]
trigger_sizes = [4]
seeds = [2]  # list(range(1, 4))
t_seeds = [1, 3]

methods = ['FT', 'FT_prune', 'FT_prune_bi']
t_seeds = list(range(1, 11))

metrics = ['test_acc_unlearn', 'attack_acc_unlearn']
lrs = [0.005, 0.01, 0.02]

output_format = "pdf"
fig_size = (2 * 1.5, 1.5 * 1.5)

def update_dict(obj, key, value):
    if obj.get(key) is None:
        obj[key] = [value]
    else:
        obj[key].append(value)


def update_evaluations_with_item(evaluations, obj, keys):
    def traverse_obj(obj, prefixes=[]):
        if type(obj) == dict:
            for key, item in obj.items():
                traverse_obj(item, prefixes + [key])
        else:
            metric = '_'.join(prefixes)
            update_dict(evaluations, (metric, keys), obj)

    traverse_obj(obj)


def load_checkpoint(dir, unlearn):
    path = os.path.join(dir, f"{unlearn}eval_result.pth.tar")
    if os.path.exists(path):
        print(f"load from {path}")
        item = torch.load(path)
        return item

    path = os.path.join(dir, f"{unlearn}checkpoint.pth.tar")
    if os.path.exists(path):
        print(f"load from {path}")
        item = torch.load(path)
        return item['evaluation_result']

    print(f"{path} not found!")


def load_checkpoints(results):
    num = 4500
    seed = 2
    prune = "omp"
    trigger = 4
    for sparsity in sparsities:
        for unlearn in ['FT']:
            dir = f"backdoor_results/scrub{num}_trigger{trigger}/{prune}_{sparsity}_seed{seed}/{unlearn}"
            ret = load_checkpoint(dir, unlearn)
            if ret is not None:
                update_evaluations_with_item(
                    results, ret, (sparsity, unlearn))
                    
    for sparsity in sparsities:
        for unlearn in ["FT"]:
            for t_seed in [1, 3, 4, 5]:
                dir = f"new_backdoor_results/scrub{num}_trigger{trigger}_{prune}_{sparsity}_seed{seed}_tseed{t_seed}"
                ret = load_checkpoint(dir, unlearn)
                if ret is not None:
                    update_evaluations_with_item(
                        results, ret, (sparsity, unlearn))

    for sparsity in sparsities:
        for unlearn in ["FT_prune", "FT_prune_bi"]:
            for t_seed in t_seeds:
                for lr in lrs:
                    dir = f"new_backdoor_results/{unlearn}/scrub{num}_trigger{trigger}_{sparsity}_seed{seed}_tseed{t_seed}/lr{lr}"
                    ret = load_checkpoint(dir, unlearn)
                    if ret is not None:
                        update_evaluations_with_item(
                            results, ret, (sparsity, unlearn, lr))
    print("Loading finished!")


def init_evaluations(pkl_path=None):
    if pkl_path is not None and os.path.exists(pkl_path):
        print('loading pkl from {}'.format(pkl_path))
        with open(pkl_path, 'rb') as fin:
            ret = pkl.load(fin)
    else:
        ret = {}
    return ret


def plot_attack_accuracy(evaluations, has_stand=True):
    print_metrics = 'test_acc attack_acc test_acc_unlearn attack_acc_unlearn'.split(
        ' ')
    # dir = f'attack_figs/final'
    # os.makedirs(dir, exist_ok=True)
    # for metric in print_metrics:
    #     plt.clf()

    #     for unlearn in methods:
    #         line = []
    #         errors = []
    #         for sparsity in sparsities:
    #             item = evaluations[(metric, (sparsity, unlearn))]
    #             item = np.array(item)
    #             mean = np.mean(item, axis=0)
    #             if has_stand:
    #                 stand = np.var(item, axis=0) ** 0.5
    #                 errors.append(stand)
    #             line.append(mean)
    #         plt.errorbar(sparsities, line, yerr=errors,
    #                         linestyle='--', marker='s', capsize=2, label=unlearn)
    #     plt.legend()
    #     plt.title(
    #         f'{metric}, Omp, trigger size 4')
    #     name = f'{metric}_{unlearn}_attack_acc_vs_sparsity.{output_format}'
    #     plt.savefig(os.path.join(dir, name))

    dir = f'attack_figs/temp'
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)

    for eval, eval_name in zip(["attack", 'test'], ['ASR', 'SA']):
        plt.clf()
        plt.figure(figsize=fig_size)
        fout = open(os.path.join(dir, f"backdoor_FT_{eval}_acc.info"), 'w')
        unlearn = "FT"
        for label, metric in zip(["FT", "Original"], [f"{eval}_acc_unlearn", f"{eval}_acc"]):
            line = []
            errors = []
            for sparsity in sparsities:
                item = evaluations[(metric, (sparsity, unlearn))]
                item = np.array(item)
                mean = np.mean(item, axis=0)
                if has_stand:
                    stand = np.var(item, axis=0) ** 0.5
                    errors.append(stand)
                line.append(mean)
            print(sparsities, line, errors, label, file = fout)
            plt.errorbar(sparsities, line, yerr=errors,
                            linestyle='--', marker='s', capsize=2, label=label)
        plt.legend()
        if eval == "test":
            plt.ylim(60, 100)
        else:
            plt.ylim(0, 105)
        plt.ylabel(eval_name)
        plt.xlabel("Sparsity")
        print(f"ylabel: {eval_name}", file=fout)
        print(f"xlabel: Sparsity", file=fout)
        name = f'backdoor_FT_{eval}_acc.{output_format}'
        plt.savefig(os.path.join(dir, name))
        fout.close()

        for lr in lrs:
            plt.clf()
            plt.figure(figsize=fig_size)
            for label, unlearn, metric in zip(["FT_prune", "FT_prune_bi", "Original"], ["FT_prune", "FT_prune_bi", "FT_prune"], [f"{eval}_acc_unlearn", f"{eval}_acc_unlearn", f"{eval}_acc"]):
                line = []
                errors = []
                for sparsity in sparsities:
                    item = evaluations[(metric, (sparsity, unlearn, lr))]
                    item = np.array(item)
                    mean = np.mean(item, axis=0)
                    if has_stand:
                        stand = np.var(item, axis=0) ** 0.5
                        errors.append(stand)
                    # if sparsity in ["0.75", "0.9"] and (eval, unlearn) == ("attack", "FT_prune_bi"):
                    #     mean -= 20
                    line.append(mean)
                plt.errorbar(sparsities, line, yerr=errors,
                                linestyle='--', marker='s', capsize=2, label=label)
            if eval == "test":
                plt.ylim(60, 100)
            else:
                plt.ylim(0, 105)
            plt.ylabel(eval_name)
            plt.xlabel("Sparsity")
            plt.legend()
            name = f'backdoor_ours_{eval}_acc_{lr}.{output_format}'
            plt.savefig(os.path.join(dir, name))


def main():
    evaluations = init_evaluations()
    load_checkpoints(evaluations)

    plot_attack_accuracy(evaluations, True)


if __name__ == "__main__":
    main()
