import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments')

    ##################################### Dataset #################################################
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--input_size', type=int, default=32, help='size of input images')
    parser.add_argument('--data_dir', type=str, default='../data/tiny-imagenet-200', help='dir to tiny-imagenet')
    parser.add_argument('--num_workers', type=int, default=4)

    ##################################### Architecture ############################################
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

    ##################################### General setting ############################################
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
    parser.add_argument('--mask', type=str, default=None, help='sparse model')

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=160, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')

    ##################################### Pruning setting #################################################
    parser.add_argument('--pruning_times', default=1, type=int, help='overall times of pruning')
    parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')  # pruning rate is always 20%
    parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
    parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
    parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')

    ##################################### unlearn setting #################################################
    parser.add_argument('--unlearn', type=str, default='retrain', help='methods to unlearn')
    parser.add_argument('--num_indexes_to_replace', type=int, default=None,
                        help='Number of samples of class to forget')
    parser.add_argument('--class_to_replace', type=int, default=0,
                        help='Class to forget')
    parser.add_argument('--indexes_to_replace', type=list, default=None,
                        help='Class to forget')
    return parser.parse_args()