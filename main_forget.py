'''
    main process for a Lottery Tickets experiments
'''
import os
import numpy as np
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from collections import OrderedDict

import utils
import unlearn
from pruner import *
from metrics import efficacy, MIA

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
    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.cuda()
    def replace_loader_dataset(data_loader, dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        loader_args = {'num_workers': 0, 'pin_memory': False}
        def _init_fn(worker_id):
            np.random.seed(int(seed))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=shuffle)

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = - forget_dataset.targets[marked] - 1
    forget_loader = replace_loader_dataset(train_loader_full, forget_dataset, seed=seed, shuffle=True)
    print(len(forget_dataset))
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = replace_loader_dataset(train_loader_full, retain_dataset, seed=seed, shuffle=True)
    print(len(retain_dataset))
    assert(len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset))

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader,
        forget=forget_loader,
        val=val_loader,
        test=test_loader)

    criterion = nn.CrossEntropyLoss()

    if args.resume:
        print('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location = device)
        # best_sa = checkpoint['best_sa']
        # start_epoch = checkpoint['epoch']
        # all_result = checkpoint['result']
        # start_state = checkpoint['state']

        # if start_state > 0:
        #     current_mask = extract_mask(checkpoint['state_dict'])
        #     prune_model_custom(model, current_mask)
        #     check_sparsity(model)
        #     optimizer = torch.optim.SGD(model.parameters(), args.lr,
        #                                 momentum=args.momentum,
        #                                 weight_decay=args.weight_decay)
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        # # adding an extra forward process to enable the masks
        # x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
        # model.eval()
        # with torch.no_grad():
        #     model(x_rand)

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # initalization = checkpoint['init_weight']
        # print('loading state:', start_state)
        # print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)

    else:
        checkpoint = torch.load(args.mask, map_location = device)
        current_mask = extract_mask(checkpoint)
        prune_model_custom(model, current_mask)
        check_sparsity(model)

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)

    forget_len = len(forget_dataset)
    retain_dataset_train = torch.utils.data.Subset(retain_dataset,list(range(len(retain_dataset)-forget_len)))
    retain_dataset_test = torch.utils.data.Subset(retain_dataset,list(range(len(retain_dataset)-forget_len,len(retain_dataset))))
    retain_loader_train = torch.utils.data.DataLoader(retain_dataset_train,batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    retain_loader_test = torch.utils.data.DataLoader(retain_dataset_test,batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print(len(retain_dataset_train))
    print(len(retain_dataset_test))
    MIA(retain_loader_train,retain_loader_test,forget_loader,test_loader,model)

    print(efficacy(model,forget_loader, device))

if __name__ == '__main__':
    main()
