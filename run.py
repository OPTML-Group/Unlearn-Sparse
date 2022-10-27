from utils import run_commands

params = {
    "FT": "--epoch 10 --lr 0.01",
    "GA": "--epoch 10 --lr 0.01",
    "RL": "--epoch 10 --lr 0.01"
}


def gen_commands(rerun = False):
    commands = []
    sparsities = "0p5 0p75 0p9 0p99 0p95 0p995".split(' ')
    methods = "FT GA RL".split(' ')  # GA RL  raw fisher retrain
    nums = [100, 4500]# , 2250, 450]
    seeds = [1, 2, 4]
    for seed in seeds:
        for num in nums:
            for unlearn in methods:
                command = f"python -u main_forget.py --save_dir unlearn_results/dense/{unlearn}_{num}/seed{seed} --mask pruning_models/dense/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed}"
                if unlearn in params:
                    command = command + ' ' + params[unlearn]
                if not rerun:
                    command = command + ' --resume'
                commands.append(command)
        for sparsity in sparsities:
            for num in nums:
                for unlearn in methods:
                    command = f"python -u main_forget.py --save_dir unlearn_results/{sparsity}/{unlearn}_{num}/seed{seed} --mask pruning_models/synflow_iterative/ratio{sparsity}/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed}"
                    if unlearn in params:
                        command = command + ' ' + params[unlearn]
                    if not rerun:
                        command = command + ' --resume'
                    commands.append(command)
    return commands


if __name__ == "__main__":
    commands = gen_commands(rerun = True)
    print(len(commands))
    run_commands(list(range(8)), commands, call=True,
                 dir="commands", shuffle=True, delay=0.5)
