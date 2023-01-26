from utils import run_commands

params = {
    "FT": "--unlearn_epochs 10 --unlearn_lr 0.01",
    "GA": "--unlearn_epochs 5 --unlearn_lr 0.001",
    "RL": "--unlearn_epochs 10 --unlearn_lr 0.01",
    "retrain": "--unlearn_epochs 160",
    "fisher_new": "--alpha 3e-8 --no-aug",
    "fisher": "--alpha 1e-6 --no-aug",
    "wfisher": "--alpha 1 --no-aug",
    "FT_prune": "--alpha 0.0001 --unlearn_epochs 10 ",#--unlearn_lr 0.01",
    "FT_prune_bi": "--unlearn_epochs 10 ",#--unlearn_lr 0.01",
}
mask_format = {
    "SynFlow": "pruning_models/synflow_iterative/ratio{sparsity}/seed1/state_dict.pth",
    "OMP": "pruning_models/OMP/Omp_resnet18_cifar10_seed1_rate_{sparsity}/1checkpoint.pth.tar",
    "IMP": "pruning_models/IMP/ratio{sparsity}/model_SA_best.pth.tar",
}


def gen_commands_unlearn(rerun=False, dense=False):
    pruning_methods = ["IMP"]  # , "OMP"]
    commands = []
    sparsities = "0.5 0.75 0.9 0.95 0.99 0.995".split(
        ' ')  # "0.5 0.75 0.9 0.95 0.99 0.995".split(' ')
    methods = "fisher_new FT RL raw retrain".split(
        ' ')  # fisher_new FT RL raw retrain wfisher
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
    methods = "ihpv_woodfisher".split(' ')  # fisher FT GA RL raw retrain
    nums = [100, 450, 2250, 4500]
    alphas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    for seed in [1, 2, 3]:
        for alpha in alphas:
            for num in nums:
                for unlearn in methods:
                    command = f"python -u main_forget.py --save_dir unlearn_results/debug_w/{unlearn}_{num}_{alpha}_{seed} --mask pruning_models/dense/seed1/state_dict.pth --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed} --alpha {alpha} --no-aug"
                    commands.append(command)
    return commands


def gen_commands_backdoor(rerun=False):
    commands = []
    pruning_methods = ["omp"]  # "synflow",
    sparsities = "0 0.5 0.75 0.9 0.95 0.99".split(' ')
    methods = "FT".split(' ')  # fisher_new FT RL raw retrain wfisher
    nums = [4500]  # , 450, 2250]
    trigger_sizes = [4]#[6, 4, 2]  # 4, 2
    seeds = [2]
    train_seeds = [1, 3]

    # for prune in pruning_methods:
    #     for sparsity in sparsities:
    #         for trigger in trigger_sizes:
    #             for num in nums:
    #                 for unlearn in methods:
    #                     for seed in seeds:
    #                         command = f"python -u main_backdoor.py --save_dir backdoor_results/scrub{num}_trigger{trigger}/{prune}_{sparsity}_seed{seed}/{unlearn} --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed} --class_to_replace 0 --prune {prune} --rate {sparsity} --trigger_size {trigger}"
    #                         if unlearn in params:
    #                             command = command + ' ' + params[unlearn]
    #                         if not rerun:
    #                             command = command + ' --resume'
    #                         commands.append(command)

    methods = ['FT_prune_bi']#, 'FT_prune']
    # train_seeds = range(6, 11)
    # for t_seed in train_seeds:
    #     for unlearn in methods:
    #         for sparsity in sparsities:
    #             for trigger in trigger_sizes:
    #                 for num in nums:
    #                     for seed in seeds:
    #                         for lr in [0.01]: #[0.02, 0.005]:
    #                             save_dir = f"new_backdoor_results/{unlearn}/scrub{num}_trigger{trigger}_{sparsity}_seed{seed}_tseed{t_seed}/lr{lr}"
    #                             mask_dir = f"new_backdoor_results/checkpoint/scrub{num}_trigger{trigger}_0_seed{seed}_tseed{t_seed}.pth"
    #                             command = f"python -u main_backdoor_alter.py --mask {mask_dir} --save_dir {save_dir} --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed} --class_to_replace 0 --rate {sparsity} --trigger_size {trigger} --train_seed {t_seed} --unlearn_lr {lr}"
    #                             if unlearn in params:
    #                                 command = command + ' ' + params[unlearn]
    #                             if not rerun:
    #                                 command = command + ' --resume'
    #                             commands.append(command)

    methods = ["FT"]
    train_seeds = [6, 11]
    for prune in pruning_methods:
        for trigger in trigger_sizes:
            for num in nums:
                for unlearn in methods:
                    for seed in seeds:
                        for sparsity in sparsities:
                            for t_seed in train_seeds:
                                save_dir = f"new_backdoor_results/scrub{num}_trigger{trigger}_{prune}_{sparsity}_seed{seed}_tseed{t_seed}"
                                mask_dir = f"new_backdoor_results/checkpoint/scrub{num}_trigger{trigger}_{sparsity}_seed{seed}_tseed{t_seed}.pth"
                                command = f"python -u main_backdoor.py --save_dir {save_dir} --mask {mask_dir} --unlearn {unlearn} --num_indexes_to_replace {num} --seed {seed} --class_to_replace 0 --prune {prune} --rate {sparsity} --trigger_size {trigger} --train_seed {t_seed}"
                                if unlearn in params:
                                    command = command + ' ' + params[unlearn]
                                if not rerun:
                                    command = command + ' --resume'
                                commands.append(command)
    return commands


def gen_commands_eigen(rerun=False):
    commands = []
    pruning_methods = ["synflow"]
    # seeds = list(range(1, 4))
    sparsities = "0.0 0.5 0.75 0.9 0.95 0.99 0.995".split(' ')
    # for prune in pruning_methods:
    #     for seed in seeds:
    #         for sparsity in sparsities:
    #             command = f"python main_eigen.py --mask pruning_models/new_pruning_models/cifar10/resnet/{prune}/ratio{sparsity}/seed{seed}/model_SA_best.pth.tar --save_dir eigen_results/{prune}_{sparsity}_seed{seed} --seed {seed}"
    #             commands.append(command)
    for sparsity in sparsities:
        for epoch in range(0, 181, 20):
            command = f"python main_eigen.py --mask pruning_models/omp_new/Omp_resnet18_cifar10_seed1_rate_{sparsity}_tj/epoch_{epoch}_weight.pt --save_dir eigen_results/single/{sparsity}_epoch{epoch}"
            commands.append(command)
    return commands


if __name__ == "__main__":
    # commands = gen_commands_unlearn(rerun=False, dense=False)
    # print(len(commands))
    # run_commands(list(range(8)) * 4, commands, call=True,
    #              dir="commands_RL", shuffle=False, delay=0.5)

    # commands = gen_commands_debug_fisher()
    # print(len(commands))
    # run_commands(list(range(0, 8)) * 3, commands, call=True,
    #              dir="commands", shuffle=False, delay=0.5)

    commands = gen_commands_backdoor(rerun=True)
    print(len(commands))
    run_commands(list(range(8)) * 3 + [0, 1, 2], commands, call=True,
                 dir="commands_attack", shuffle=False, delay=0.5)

    # commands = gen_commands_eigen(rerun=False)
    # print(len(commands))
    # run_commands(list(range(8)) * 1, commands, call=True,
    #              dir="commands_eigen", shuffle=False, delay=0.5)
