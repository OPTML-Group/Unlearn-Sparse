import torch
import os
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import shutil

sparsities = "0.0 0.5 0.75 0.9 0.95 0.99 0.995".split(' ')
methods = "raw retrain FT RL fisher_new".split(' ')  # fisher
nums = [100, 450, 2250, 4500]
seeds = list(range(1, 11))
metrics = ['accuracy_retain',
           'accuracy_forget',
           'accuracy_val',
           'accuracy_test',
           'efficacy',
           'SVC_MIA_forget_efficacy_correctness',
           'SVC_MIA_forget_efficacy_confidence',
           'SVC_MIA_forget_efficacy_entropy',
           'SVC_MIA_training_privacy_correctness',
           'SVC_MIA_training_privacy_confidence',
           'SVC_MIA_training_privacy_entropy',
           'SVC_MIA_forget_privacy_correctness',
           'SVC_MIA_forget_privacy_confidence',
           'SVC_MIA_forget_privacy_entropy']
pruning_methods = ["SynFlow", "OMP"]
pruning_method = pruning_methods[0]


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
    for seed in seeds:
        for sparsity in sparsities:
            for num in nums:
                for unlearn in methods:
                    if sparsity == "0.0":
                        dir = f"unlearn_results/dense/{unlearn}_{num}/seed{seed}"
                    else:
                        dir = f"unlearn_results/{pruning_method}/{sparsity}/{unlearn}_{num}/seed{seed}"
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


def plot_accuracy(evaluations, fout, has_stand=True):
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
                    output = "{:.2f}".format(mean)
                    if has_stand:
                        stand = np.var(item, axis=0) ** 0.5
                        output += "±{:.2f}".format(stand)
                    line.append(output)
                print('\t'.join(x for x in line), file=fout)


def plot_MIA(evaluations, fout, has_stand=True):
    print_prefixes = 'SVC_MIA_forget_efficacy SVC_MIA_training_privacy SVC_MIA_forget_privacy'.split(
        ' ')
    print_suffixes = 'correctness confidence entropy'.split(' ')
    for pref in print_prefixes:
        for suff in print_suffixes:
            metric = f'{pref}_{suff}'
            print(metric, file=fout)
            for num in nums:
                print(f"scrub {num}:", file=fout)
                for unlearn in methods:
                    line = [unlearn]
                    for sparsity in sparsities:
                        item = evaluations[(metric, num, unlearn, sparsity)]
                        item = np.array(item) * 100
                        mean = np.mean(item, axis=0)
                        output = "{:.2f}".format(mean)
                        if has_stand:
                            stand = np.var(item, axis=0) ** 0.5
                            output += "±{:.2f}".format(stand)
                        line.append(output)
                    print('\t'.join(x for x in line), file=fout)


def plot_efficacy(evaluations):
    metric = "efficacy"
    for num in nums:
        dir = f'figs/{num}_{pruning_method}'
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

        for unlearn in methods:
            plt.clf()
            for sparsity in sparsities:
                eff = evaluations[(metric, num, unlearn, sparsity)]
                mia = evaluations[("SVC_MIA_forget_efficacy_entropy", num, unlearn, sparsity)]
                ax = sns.scatterplot(x=eff, y=mia)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Efficacy')
            ax.set_ylabel('MIA prob')
            ax.legend(sparsities)
            ax.set_title(f"Scrub {num}, {unlearn} efficacy score vs sparsity")
            name = f'MIA_efficacy_vs_sparsity_{unlearn}.png'
            plt.savefig(os.path.join(dir, name))

        for unlearn in methods:
            plt.clf()
            for id, sparsity in enumerate(sparsities):
                eff = evaluations[(metric, num, unlearn, sparsity)]
                mia = evaluations[("SVC_MIA_forget_efficacy_entropy", num, unlearn, sparsity)]
                ax = sns.scatterplot(x=eff, y=[id] * len(eff))

            ax.set_xscale('log')
            ax.set_xlabel('Efficacy')
            ax.legend(sparsities)
            ax.set_title(f"Scrub {num}, {unlearn} efficacy score vs sparsity")
            name = f'grouped_efficacy_vs_sparsity_{unlearn}.png'
            plt.savefig(os.path.join(dir, name))

        # efficacy dist vs method
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

        for sparsity in sparsities:
            plt.clf()
            for unlearn in methods:
                eff = evaluations[(metric, num, unlearn, sparsity)]
                mia = evaluations[("SVC_MIA_forget_efficacy_entropy", num, unlearn, sparsity)]
                ax = sns.scatterplot(x=eff, y=mia)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Efficacy')
            ax.set_ylabel('MIA prob')
            ax.legend(methods)
            ax.set_title(
                f"Scrub {num}, {sparsity} sparsity efficacy score vs unlearn methods")
            name = f'MIA_efficacy_vs_methods_{sparsity}.png'
            plt.savefig(os.path.join(dir, name))

        for sparsity in sparsities:
            plt.clf()
            for id, unlearn in enumerate(methods):
                eff = evaluations[(metric, num, unlearn, sparsity)]
                mia = evaluations[("SVC_MIA_forget_efficacy_entropy", num, unlearn, sparsity)]
                ax = sns.scatterplot(x=eff, y=[id] * len(eff))

            ax.set_xscale('log')
            ax.set_xlabel('Efficacy')
            ax.legend(methods)
            ax.set_title(
                f"Scrub {num}, {sparsity} sparsity efficacy score vs unlearn methods")
            name = f'grouped_efficacy_vs_methods_{sparsity}.png'
            plt.savefig(os.path.join(dir, name))


def main():
    evaluations = init_evaluations()  # f"export_{pruning_method}_12345.pkl")
    load_checkpoints(evaluations)

    # export_path = "export_{}_{}.pkl".format(
    #     pruning_method, ''.join(str(x) for x in seeds))
    # with open(export_path, 'wb') as fout:
    #     pkl.dump(evaluations, fout)
    # from IPython import embed
    # embed()

    for has_stand in [True, False]:
        log_name = f"output_{pruning_method}.log"
        if not has_stand:
            log_name = f"output_no_stand_{pruning_method}.log"

        with open(log_name, 'w') as fout:
            plot_accuracy(evaluations, fout, has_stand)
            plot_MIA(evaluations, fout, has_stand)

    # shutil.rmtree("figs", ignore_errors=True)
    plot_efficacy(evaluations)


if __name__ == "__main__":
    main()
