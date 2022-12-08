from utils import run_commands

params = {
    "FT": "--epoch 10 --lr 0.01",
    "GA": "--epoch 5 --lr 0.001",
    "RL": "--epoch 10 --lr 0.01",
    "fisher_new": "--alpha 3e-8 --no-aug",
    "fisher": "--alpha 1e-6 --no-aug"
}
mask_format = {
    "SynFlow": "pruning_models/synflow_iterative/ratio{sparsity}/seed1/state_dict.pth",
    # "OMP": "pruning_models/OMP/Omp_resnet18_cifar10_seed1_rate_{sparsity}/1checkpoint.pth.tar"
}


def gen_commands_unlearn(rerun=False):
    pruning_methods = ["SynFlow"]#, "OMP"]
    commands = []
    sparsities = "0.5 0.75 0.9 0.95 0.99 0.995".split(' ')
    methods = "FT GA RL fisher_new retrain".split(' ')  # fisher_new FT GA RL raw retrain
    nums = [4500]
    seeds = [1,2,3,4,5]

    # dense
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
    methods = "fisher_new".split(' ')  # fisher FT GA RL raw retrain
    nums = [100, 2250, 450, 4500]
    alphas = [2e-8, 22e-9, 24e-9, 25e-9, 26e-9, 28e-9, 3e-8]
    for seed in [1,2,3,4,5]:
        for alpha in alphas:
            for num in nums:
                for unlearn in methods:
                    command = f"python -u main_forget.py --save_dir unlearn_results/debug_32/{unlearn}_{num}_{alpha}_{seed} --mask pruning_models/dense/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed} --alpha {alpha} --no-aug"
                    commands.append(command)
    return commands


if __name__ == "__main__":
    commands = gen_commands_unlearn(rerun=True)
    print(len(commands))
    run_commands(list(range(8)) * 3, commands, call=True,
                 dir="commands", shuffle=False, delay=0.5)
    # commands = gen_commands_debug_fisher()
    # print(len(commands))
    # run_commands(list(range(0, 8)) * 3, commands, call=True,
    #              dir="commands", shuffle=False, delay=0.5)
