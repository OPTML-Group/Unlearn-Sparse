import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )
    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=0, help="Specific class to forget"
    )
    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")

    ##################################### Attack setting #################################################
    parser.add_argument(
        "--attack", type=str, default="backdoor", help="method to unlearn"
    )
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=4,
        help="The size of trigger of backdoor attack",
    )
    ##################################### Transfer learning setting #####################################
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default=None,
        help="ImageNet source/unlearned checkpoint for transfer learning",
    )
    parser.add_argument(
        "--source_num_classes",
        type=int,
        default=1000,
        help="Number of classes in the source checkpoint head",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="oxfordpets",
        help="Downstream transfer dataset: oxfordpets, sun397, or imagefolder",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default="./data",
        help="Root directory for downstream transfer datasets",
    )
    parser.add_argument(
        "--target_train_path",
        type=str,
        default=None,
        help="Optional ImageFolder train path for transfer learning",
    )
    parser.add_argument(
        "--target_test_path",
        type=str,
        default=None,
        help="Optional ImageFolder test path for transfer learning",
    )
    parser.add_argument(
        "--transfer_split_file",
        type=str,
        default=None,
        help="Optional CoOp-style JSON split file with train/val/test entries",
    )
    parser.add_argument(
        "--transfer_include_val",
        action="store_true",
        help="Include val entries from --transfer_split_file in transfer training",
    )
    parser.add_argument(
        "--transfer_train_ratio",
        type=float,
        default=0.8,
        help="Deterministic train ratio for datasets without official train/test split",
    )
    parser.add_argument(
        "--transfer_method",
        type=str,
        default="lp",
        choices=["lp", "ff"],
        help="Transfer method: lp freezes the feature extractor, ff full-finetunes",
    )
    parser.add_argument(
        "--transfer_epochs", type=int, default=200, help="Transfer training epochs"
    )
    parser.add_argument(
        "--transfer_lr", type=float, default=1e-4, help="Transfer optimizer learning rate"
    )
    parser.add_argument(
        "--transfer_weight_decay",
        type=float,
        default=0.0,
        help="Transfer optimizer weight decay",
    )
    parser.add_argument(
        "--transfer_optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD"],
        help="Transfer optimizer",
    )
    parser.add_argument(
        "--transfer_resolution",
        type=int,
        default=224,
        help="Input resolution for downstream transfer learning",
    )
    parser.add_argument(
        "--transfer_test_batch_size",
        type=int,
        default=1024,
        help="Transfer evaluation batch size",
    )
    parser.add_argument(
        "--transfer_download",
        action="store_true",
        help="Download torchvision downstream datasets when missing",
    )
    parser.add_argument(
        "--transfer_eval_only",
        action="store_true",
        help="Only evaluate a transfer checkpoint from --checkpoint",
    )
    return parser.parse_args(argv)
