import torch
import os
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import shutil


pruning_methods = ['omp']  # ["synflow", "omp"]
sparsities = "0 0.5 0.75 0.9 0.95 0.99".split(' ')
methods = "FT".split(' ')  # fisher_new FT RL raw retrain wfisher
nums = [4500]  # [450, 2250, 4500]
trigger_sizes = [2, 4, 6]
seeds = [2]  # list(range(1, 4))
t_seeds = [1, 3]

methods = ['FT_prune', 'FT_prune_bi']
t_seeds = [1, 2, 3]

metrics = ['test_acc_unlearn', 'attack_acc_unlearn']


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
    # for num in nums:
    #     for trigger in trigger_sizes:
    #         for prune in pruning_methods:
    #             for sparsity in sparsities:
    #                 for seed in seeds:
    #                     for unlearn in methods:
    #                         dir = f"backdoor_results/scrub{num}_trigger{trigger}/{prune}_{sparsity}_seed{seed}/{unlearn}"
    #                         ret = load_checkpoint(dir, unlearn)
    #                         if ret is not None:
    #                             update_evaluations_with_item(
    #                                 results, ret, ((num, trigger), (prune, sparsity, unlearn)))
    # for num in nums:
    #     for trigger in trigger_sizes:
    #         for prune in pruning_methods:
    #             for sparsity in sparsities:
    #                 for seed in seeds:
    #                     for unlearn in methods:
    #                         for t_seed in t_seeds:
    #                             dir = f"new_backdoor_results/scrub{num}_trigger{trigger}_{prune}_{sparsity}_seed{seed}_tseed{t_seed}"
    #                             ret = load_checkpoint(dir, unlearn)
    #                             if ret is not None:
    #                                 update_evaluations_with_item(
    #                                     results, ret, ((num, trigger), (prune, sparsity, unlearn)))
    for num in nums:
        for trigger in trigger_sizes:
            for prune in pruning_methods:
                for sparsity in sparsities:
                    for seed in seeds:
                        for unlearn in methods:
                            for t_seed in t_seeds:
                                dir = f"new_backdoor_results/{unlearn}/scrub{num}_trigger{trigger}_{prune}_{sparsity}_seed{seed}_tseed{t_seed}"
                                ret = load_checkpoint(dir, unlearn)
                                if ret is not None:
                                    update_evaluations_with_item(
                                        results, ret, ((num, trigger), (prune, sparsity, unlearn)))
    print("Loading finished!")


def init_evaluations(pkl_path=None):
    if pkl_path is not None and os.path.exists(pkl_path):
        print('loading pkl from {}'.format(pkl_path))
        with open(pkl_path, 'rb') as fin:
            ret = pkl.load(fin)
    else:
        ret = {}
    return ret


def print_accuracy(evaluations, fout, has_stand=True):
    print_metrics = 'test_acc attack_acc test_acc_unlearn attack_acc_unlearn'.split(
        ' ')

    for prune in pruning_methods:
        print(f"Prune: {prune}", file=fout)
        for num in nums:
            for trigger in trigger_sizes:
                print(f"scrub {num}, trigger size {trigger}:", file=fout)
                for metric in print_metrics:
                    # print(f"pruning method: {prune}", file=fout)
                    for unlearn in methods:
                        line = [metric]
                        for sparsity in sparsities:
                            item = evaluations[(
                                metric, ((num, trigger), (prune, sparsity, unlearn)))]
                            item = np.array(item)
                            mean = np.mean(item, axis=0)
                            output = "{:.2f}".format(mean)
                            if has_stand:
                                stand = np.var(item, axis=0) ** 0.5
                                output += "±{:.2f}".format(stand)
                            line.append(output)
                        print('\t'.join(x for x in line), file=fout)


def plot_attack_accuracy(evaluations, has_stand=True):
    print_metrics = 'test_acc attack_acc test_acc_unlearn attack_acc_unlearn'.split(
        ' ')

    for metric in print_metrics:
        for prune in pruning_methods:
            dir = f'attack_figs/{prune}'
            os.makedirs(dir, exist_ok=True)

            for unlearn in methods:
                for trigger in trigger_sizes:
                    plt.clf()
                    for num in nums:
                        line = []
                        errors = []
                        for sparsity in sparsities:
                            item = evaluations[(
                                metric, ((num, trigger), (prune, sparsity, unlearn)))]
                            item = np.array(item)
                            mean = np.mean(item, axis=0)
                            if has_stand:
                                stand = np.var(item, axis=0) ** 0.5
                                errors.append(stand)
                            line.append(mean)
                        plt.errorbar(sparsities, line, yerr=errors,
                                     linestyle='--', marker='s', capsize=2, label=num)
                    plt.legend()
                    plt.title(
                        f'{metric}, {prune}, trigger size {trigger}, {unlearn}')
                    name = f'{metric}_trigger{trigger}_{unlearn}_attack_acc_vs_sparsity.png'
                    plt.savefig(os.path.join(dir, name))

    # for metric in ['attack_acc_unlearn']:
    #     for prune in pruning_methods:
    #         for idx in range(3):
    #             dir = f'attack_figs/seeds_{idx + 1}'
    #             os.makedirs(dir, exist_ok=True)

    #             for unlearn in methods:
    #                 for trigger in trigger_sizes:
    #                     plt.clf()
    #                     for num in nums[1:]:
    #                         line = []
    #                         errors = []
    #                         for sparsity in sparsities:
    #                             item = evaluations[(
    #                                 metric, ((num, trigger), (prune, sparsity, unlearn)))]
    #                             line.append(float(item[idx]))
    #                         plt.plot(sparsities, line,
    #                                  linestyle='--', marker='s', label=num)
    #                     plt.legend()
    #                     plt.title(
    #                         f'{metric}, {prune}, trigger size {trigger}, {unlearn}, seed {idx + 1}')
    #                     name = f'{prune}_{metric}_trigger{trigger}_{unlearn}_attack_acc_vs_sparsity.png'
    #                     plt.savefig(os.path.join(dir, name))


def main():
    evaluations = init_evaluations()
    load_checkpoints(evaluations)

    # export_path = "export_{}_{}.pkl".format(
    #     pruning_method, ''.join(str(x) for x in seeds))
    # with open(export_path, 'wb') as fout:
    #     pkl.dump(evaluations, fout)
    # from IPython import embed
    # embed()

    for has_stand in [True, False]:
        log_name = f"output_attack.log"
        if not has_stand:
            log_name = f"output_attack_no_stand.log"

        with open(log_name, 'w') as fout:
            print_accuracy(evaluations, fout, has_stand)

    shutil.rmtree("attack_figs", ignore_errors=True)
    plot_attack_accuracy(evaluations, True)
    # plot_efficacy(evaluations)


if __name__ == "__main__":
    main()
