import torch
import os
import pickle as pkl

sparsities = "dense 0p5 0p75 0p9".split(' ')#  0p99 0p95 0p995
methods = "raw fisher retrain FT".split(' ')  # GA RL
nums = [100, 4500]# , 2250, 450]
seeds = [3, 5]
metrics = ['accuracy_retain', 'accuracy_forget', 'accuracy_val', 'accuracy_test', 'MIA_correctness', 'MIA_confidence', 'MIA_entropy', 'MIA_m_entropy', 'efficacy']

def update_dict(obj, key, value):
    if obj.get(key) is None:
        obj[key] = [value]
    else:
        obj[key].append(value)

def update_evaluations_with_item(evaluations, obj, num, unlearn, sparsity):
    def traverse_obj(obj, prefixes = []):
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
        # print(f"load from {path}")
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
                        update_evaluations_with_item(results, ret, num, unlearn, sparsity)
    print("Loading finished!")

def init_evaluations(pkl_path = None):
    if pkl_path is not None and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as fin:
            ret = pkl.load(fin)
    else:
        ret = {}
    return ret


def main():
    evaluations = init_evaluations()

    load_checkpoints(evaluations)

    with open("export.pkl", 'wb') as fout:
        pkl.dump(evaluations, fout)
    print_metrics = 'accuracy_retain MIA_confidence MIA_entropy efficacy'.split(' ')
    with open('output.log', 'w') as fout:
        for metric in print_metrics:
            print(metric, file = fout)
            for num in nums:
                print(f"scrub {num}:", file = fout)
                for unlearn in methods:
                    line = []
                    for sparsity in sparsities:
                        item = evaluations[(metric, num, unlearn, sparsity)]
                        output = str(item)
                        line.append(output)
                    print('\t'.join(x for x in line), file = fout)



if __name__ == "__main__":
    main()