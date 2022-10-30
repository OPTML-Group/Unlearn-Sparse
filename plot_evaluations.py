import torch
import os
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import shutil

sparsities = "dense 0p5 0p75 0p9 0p95 0p99 0p995".split(' ')
methods = "raw retrain FT GA RL".split(' ')  # fisher
nums = [100, 4500]  # , 2250, 450]
seeds = [1, 2, 3, 4, 5]
metrics = ['accuracy_retain', 'accuracy_forget', 'accuracy_val', 'accuracy_test', 'MIA_correctness',
           'MIA_confidence', 'MIA_entropy', 'MIA_m_entropy', 'efficacy', 'SVC_MIA', 'SVC_MIA_forget']


def update_dict(obj, key, value):
    if obj.get(key) is None:
        obj[key] = [value]
    else:
        obj[key].append(value)


def update_evaluations_with_item(evaluations, obj, num, unlearn, sparsity):
    def traverse_obj(obj, prefixes=[]):
        if type(obj) == dict:
            for key, item in obj.items():
                traverse_obj(item, prefixes + [key])
        else:
            metric = '_'.join(prefixes)
            update_dict(evaluations, (metric, num, unlearn, sparsity), obj)

    traverse_obj(obj)


def load_checkpoint(dir, unlearn):
    path = os.path.join(dir, f"{unlearn}checkpoint.pth.tar")
    if os.path.exists(path):
        print(f"load from {path}")
        item = torch.load(path)
        return item['evaluation_result']

    print(f"{path} not found!")


def load_checkpoints(results):
    for seed in seeds:
        for sparsity in sparsities:
            for num in nums:
                for unlearn in methods:
                    dir = f"unlearn_results/{sparsity}/{unlearn}_{num}/seed{seed}"
                    ret = load_checkpoint(dir, unlearn)
                    if ret is not None:
                        update_evaluations_with_item(
                            results, ret, num, unlearn, sparsity)
    print("Loading finished!")


def init_evaluations(pkl_path=None):
    if pkl_path is not None and os.path.exists(pkl_path):
        print('loading pkl from {}'.format(pkl_path))
        with open(pkl_path, 'rb') as fin:
            ret = pkl.load(fin)
    else:
        ret = {}
    return ret


def plot_accuracy(evaluations, fout):
    print_metrics = 'accuracy_retain accuracy_forget accuracy_test'.split(' ')
    for metric in print_metrics:
        print(metric, file=fout)
        for num in nums:
            print(f"scrub {num}:", file=fout)
            for unlearn in methods:
                line = [unlearn]
                for sparsity in sparsities:
                    item = evaluations[(metric, num, unlearn, sparsity)]
                    item = np.array(item)
                    mean = np.mean(item, axis=0)
                    norm = np.var(item, axis=0) ** 0.5

                    output = "{:.2f}±{:.2f}".format(mean, norm)
                    line.append(output)
                print('\t'.join(x for x in line), file=fout)


def plot_MIA(evaluations, fout):
    print_metrics = 'SVC_MIA'.split(' ')
    for metric in print_metrics:
        for i in range(2):
            if i == 0:
                suffix = "_retain"
            else:
                suffix = "_forget"
            print(metric + suffix, file=fout)
            for num in nums:
                print(f"scrub {num}:", file=fout)
                for unlearn in methods:
                    line = [unlearn]
                    for sparsity in sparsities:
                        item = evaluations[(metric, num, unlearn, sparsity)]
                        item = np.array(item) * 100
                        mean = np.mean(item, axis=0)
                        norm = np.var(item, axis=0) ** 0.5

                        output = "{:.2f}±{:.2f}".format(mean[i], norm[i])
                        line.append(output)
                    print('\t'.join(x for x in line), file=fout)


def plot_MIA_forget(evaluations, fout):
    print_metrics = 'SVC_MIA_forget'.split(' ')
    for metric in print_metrics:
        print(metric, file=fout)
        for num in nums:
            print(f"scrub {num}:", file=fout)
            for unlearn in methods:
                line = [unlearn]
                for sparsity in sparsities:
                    item = evaluations[(metric, num, unlearn, sparsity)]
                    item = (np.array(item) * 100).mean(axis=1)
                    mean = np.mean(item, axis=0)
                    norm = np.var(item, axis=0) ** 0.5

                    output = "{:.2f}±{:.2f}".format(mean, norm)
                    line.append(output)
                print('\t'.join(x for x in line), file=fout)


def plot_efficacy(evaluations):
    metric = "efficacy"
    for num in nums:
        dir = f'figs/{num}'
        os.makedirs(dir, exist_ok=True)

        # efficacy dist vs sparsity
        for unlearn in methods:
            plt.clf()
            for sparsity in sparsities:
                eff = evaluations[(metric, num, unlearn, sparsity)]
                ax = sns.distplot(x=eff, hist=False)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Efficacy')
            ax.legend(sparsities)
            ax.set_title(f"Scrub {num}, {unlearn} efficacy score vs sparsity")
            name = f'efficacy_vs_sparsity_{unlearn}.png'
            plt.savefig(os.path.join(dir, name))

        # efficacy dist vs sparsity
        for sparsity in sparsities:
            plt.clf()
            for unlearn in methods:
                eff = evaluations[(metric, num, unlearn, sparsity)]
                ax = sns.distplot(x=eff, hist=False)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Efficacy')
            ax.legend(methods)
            ax.set_title(
                f"Scrub {num}, {sparsity} sparsity efficacy score vs unlearn methods")
            name = f'efficacy_vs_methods_{sparsity}.png'
            plt.savefig(os.path.join(dir, name))


def main():
    evaluations = init_evaluations()

    load_checkpoints(evaluations)
    export_path = "export_{}.pkl".format(''.join(str(x) for x in seeds))

    with open(export_path, 'wb') as fout:
        pkl.dump(evaluations, fout)

    with open('output.log', 'w') as fout:
        plot_accuracy(evaluations, fout)
        plot_MIA(evaluations, fout)
        plot_MIA_forget(evaluations, fout)
    shutil.rmtree("figs", ignore_errors=True)
    plot_efficacy(evaluations)


if __name__ == "__main__":
    main()
