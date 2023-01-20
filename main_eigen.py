import os
import torch
import torch.optim
import torch.utils.data

import utils
import pruner
import evaluation

import arg_parser
import pickle as pkl

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

    checkpoint = torch.load(args.mask, map_location=device)
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    current_mask = pruner.extract_mask(checkpoint)
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    eigens, weights = evaluation.lanczos(model, train_loader_full, 90)
    print(eigens.shape)
    path = os.path.join(args.save_dir, 'eigen_90.pkl')

    with open(path, 'wb') as fpkl:
        pkl.dump((eigens.cpu().numpy(), weights.cpu().numpy()), fpkl)


if __name__ == '__main__':
    main()
