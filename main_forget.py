import os
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from collections import OrderedDict

import utils
import unlearn
import pruner
from trainer import validate
import evaluation

import arg_parser

best_sa = 0


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
    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(
        args)
    model.cuda()

    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle)

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = - forget_dataset.targets[marked] - 1
    forget_loader = replace_loader_dataset(
        forget_dataset, seed=seed, shuffle=True)
    print(len(forget_dataset))
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = replace_loader_dataset(
        retain_dataset, seed=seed, shuffle=True)
    print(len(retain_dataset))
    assert(len(forget_dataset) + len(retain_dataset)
           == len(train_loader_full.dataset))

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader,
        forget=forget_loader,
        val=val_loader,
        test=test_loader)

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
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

    if 'accuracy' not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result['accuracy'] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if 'MIA' not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset)
        utils.dataset_convert_to_test(forget_loader)
        utils.dataset_convert_to_test(test_loader)


        retain_dataset_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        retain_dataset_test = torch.utils.data.Subset(retain_dataset, list(range(retain_len-forget_len, retain_len)))
        retain_loader_train = torch.utils.data.DataLoader(
            retain_dataset_train, batch_size=args.batch_size, shuffle=False)
        retain_loader_test = torch.utils.data.DataLoader(
            retain_dataset_test, batch_size=args.batch_size, shuffle=False)

        print(len(retain_dataset_train))
        print(len(retain_dataset_test))

        evaluation_result['MIA'] = evaluation.MIA(
            retain_loader_train, retain_loader_test, forget_loader, test_loader, model, device)
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if 'SVC_MIA' not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset)
        utils.dataset_convert_to_test(forget_loader)
        utils.dataset_convert_to_test(test_loader)


        retain_dataset_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        retain_dataset_test = torch.utils.data.Subset(retain_dataset, list(range(retain_len-forget_len, retain_len)))
        retain_loader_train = torch.utils.data.DataLoader(
            retain_dataset_train, batch_size=args.batch_size, shuffle=False)
        retain_loader_test = torch.utils.data.DataLoader(
            retain_dataset_test, batch_size=args.batch_size, shuffle=False)

        print(len(retain_dataset_train))
        print(len(retain_dataset_test))

        evaluation_result['SVC_MIA'] = evaluation.SVC_MIA(
            retain_loader_train, retain_loader_test, forget_loader, test_loader, model, device)
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if 'efficacy' not in evaluation_result:
        utils.dataset_convert_to_test(forget_loader.dataset)
        efficacy_score = evaluation.efficacy(model, forget_loader, device)
        print('efficacy score: {}'.format(efficacy_score))

        evaluation_result['efficacy'] = efficacy_score
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == '__main__':
    main()
