from utils import run_commands


def gen_commands():
    commands = []
    sparsities = "0p5 0p75 0p9 0p99 0p95 0p995".split(' ')
    methods = "raw fisher retrain FT RL".split(' ') # GA
    nums = [100, 4500, 2250, 450]
    seeds = [1, 2, 3]
    for seed in seeds:
        for num in nums:
            for unlearn in methods:
                command = f"python -u main_forget.py --save_dir unlearn_results/dense/{unlearn}_{num}/seed{seed} --mask pruning_models/dense/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed}"
                commands.append(command)
        for sparsity in sparsities:
            for num in nums:
                for unlearn in methods:
                    command = f"python -u main_forget.py --save_dir unlearn_results/{sparsity}/{unlearn}_{num}/seed{seed} --mask pruning_models/synflow_iterative/ratio{sparsity}/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed}"
                    commands.append(command)
    return commands


if __name__ == "__main__":
    commands = gen_commands()
    print(len(commands))
    run_commands([0, 1, 2, 3, 4, 5, 6, 7, 1, 2], commands, call=False,
                 dir="commands", shuffle=False, delay=0.5)
