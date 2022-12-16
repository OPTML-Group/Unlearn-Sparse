from utils import run_commands

params = {
    "FT": "--epoch 10 --lr 0.01",
    "GA": "--epoch 5 --lr 0.001",
    "RL": "--epoch 10 --lr 0.01",
    "fisher_new": "--alpha 3e-8 --no-aug",
    "fisher": "--alpha 1e-6 --no-aug",
    "wfisher": "--alpha 1 --no-aug"
}
mask_format = {
    "SynFlow": "pruning_models/synflow_iterative/ratio{sparsity}/seed1/state_dict.pth",
    "OMP": "pruning_models/OMP/Omp_resnet18_cifar10_seed1_rate_{sparsity}/1checkpoint.pth.tar"
}


def gen_commands_unlearn(rerun=False, dense=False):
    pruning_methods = ["SynFlow"]  # , "OMP"]
    commands = []
    sparsities = "0.5 0.75 0.9 0.95 0.99 0.995".split(' ')# "0.5 0.75 0.9 0.95 0.99 0.995".split(' ')
    methods = "wfisher".split(' ')  # fisher_new FT RL raw retrain
    nums = [100, 4500, 2250, 450]
    seeds = list(range(1, 6))

    # dense
    if dense:
        for seed in seeds:
            for num in nums:
                for unlearn in methods:
                    command = f"python -u main_forget.py --save_dir unlearn_results/dense/{unlearn}_{num}/seed{seed} --mask pruning_models/dense/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed}"
                    if unlearn in params:
                        command = command + ' ' + params[unlearn]
                    if not rerun:
                        command = command + ' --resume'
                    commands.append(command)

    # pruned
    for pruning_method in pruning_methods:
        for seed in seeds:
            for sparsity in sparsities:
                for num in nums:
                    for unlearn in methods:
                        mask_path = mask_format[pruning_method].format(
                            sparsity=sparsity)
                        command = f"python -u main_forget.py --save_dir unlearn_results/{pruning_method}/{sparsity}/{unlearn}_{num}/seed{seed} --mask {mask_path} --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed}"
                        if unlearn in params:
                            command = command + ' ' + params[unlearn]
                        if not rerun:
                            command = command + ' --resume'
                        commands.append(command)
    return commands


def gen_commands_debug_fisher():
    commands = []
    methods = "wfisher".split(' ')  # fisher FT GA RL raw retrain
    nums = [100, 450, 2250, 4500]
    alphas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    for seed in [1, 2, 3]:
        for alpha in alphas:
            for num in nums:
                for unlearn in methods:
                    command = f"python -u main_forget.py --save_dir unlearn_results/debug_w/{unlearn}_{num}_{alpha}_{seed} --mask pruning_models/dense/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed} --alpha {alpha} --no-aug"
                    commands.append(command)
    return commands


if __name__ == "__main__":
    # commands = gen_commands_unlearn(rerun=False, dense=True)
    # print(len(commands))
    # run_commands(list(range(8)) * 4, commands, call=False,
    #              dir="commands_RL", shuffle=False, delay=0.5)
    commands = gen_commands_debug_fisher()
    print(len(commands))
    run_commands(list(range(0, 8)) * 3, commands, call=True,
                 dir="commands", shuffle=False, delay=0.5)
