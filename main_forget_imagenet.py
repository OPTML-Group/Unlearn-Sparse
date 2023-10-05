import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import evaluation
import pruner
import unlearn
import utils
from imagenet import get_x_y_from_data_dict
from trainer import validate


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
    print(len(retain_loader.dataset))
    print(len(forget_loader.dataset))
    model.cuda()
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=val_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        current_mask = pruner.extract_mask(checkpoint)
        pruner.prune_model_custom(model, current_mask)
        pruner.check_sparsity(model)

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "accuracy" not in evaluation_result:
        accuracy = {}
        unlearn_data_loaders = dict(reversed(list(unlearn_data_loaders.items())))
        for name, loader in unlearn_data_loaders.items():
            print("start testing")
            # utils.dataset_convert_to_test(loader.dataset,args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    # if 'new_accuracy' not in evaluation_result:
    #     accuracy = {}
    #     for name, loader in unlearn_data_loaders.items():
    #         print("start testing")
    #         # utils.dataset_convert_to_test(loader.dataset,args)
    #         val_acc = validate(loader, model, criterion, args)
    #         accuracy[name] = val_acc
    #         print(f"{name} acc: {val_acc}")

    #     evaluation_result['accuracy'] = accuracy
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    # for deprecated in ['MIA', 'SVC_MIA', 'SVC_MIA_forget']:
    #     if deprecated in evaluation_result:
    #         evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(val_loader.dataset)
        N = 10000
        print(test_len)
        forget_dataset = forget_loader.dataset
        retain_dataset = retain_loader.dataset
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)
        val_dataset = torch.utils.data.Subset(val_loader.dataset, list(range(N)))
        # utils.dataset_convert_to_test(retain_dataset,args)
        # utils.dataset_convert_to_test(forget_loader,args)
        # utils.dataset_convert_to_test(test_loader,args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(N)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=val_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    """training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)"""
    # if 'SVC_MIA_training_privacy' not in evaluation_result:
    #     test_len = len(test_loader.dataset)
    #     retain_len = len(retain_dataset)
    #     num = test_len // 2

    #     # utils.dataset_convert_to_test(retain_dataset,args)
    #     # utils.dataset_convert_to_test(forget_loader,args)
    #     # utils.dataset_convert_to_test(test_loader,args)

    #     shadow_train = torch.utils.data.Subset(
    #         retain_dataset, list(range(num)))
    #     target_train = torch.utils.data.Subset(
    #         retain_dataset, list(range(num, retain_len)))
    #     shadow_test = torch.utils.data.Subset(
    #         test_loader.dataset, list(range(num)))
    #     target_test = torch.utils.data.Subset(
    #         test_loader.dataset, list(range(num, test_len)))

    #     shadow_train_loader = torch.utils.data.DataLoader(
    #         shadow_train, batch_size=args.batch_size, shuffle=False)
    #     shadow_test_loader = torch.utils.data.DataLoader(
    #         shadow_test, batch_size=args.batch_size, shuffle=False)

    #     target_train_loader = torch.utils.data.DataLoader(
    #         target_train, batch_size=args.batch_size, shuffle=False)
    #     target_test_loader = torch.utils.data.DataLoader(
    #         target_test, batch_size=args.batch_size, shuffle=False)

    #     evaluation_result['SVC_MIA_training_privacy'] = evaluation.SVC_MIA(
    #         shadow_train=shadow_train_loader, shadow_test=shadow_test_loader,
    #         target_train=target_train_loader, target_test=target_test_loader,
    #         model=model)
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
