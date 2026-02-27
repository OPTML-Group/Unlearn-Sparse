import argparse

import arg_parser
from forget_pipeline import run_forget
from pruning_pipeline import run_pruning


def build_cli():
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for pruning and unlearning"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prune = subparsers.add_parser("prune", help="Train + iterative pruning")
    prune.add_argument(
        "--profile",
        default="imp",
        choices=["imp", "ls", "sam", "vit"],
        help="Pruning/training profile",
    )

    unlearn = subparsers.add_parser("unlearn", help="Run unlearning")
    unlearn.add_argument(
        "--imagenet",
        action="store_true",
        help="Use ImageNet-specific unlearning pipeline",
    )
    return parser


def main():
    cli = build_cli()
    cli_args, passthrough = cli.parse_known_args()
    run_args = arg_parser.parse_args(passthrough)

    if cli_args.command == "prune":
        run_pruning(run_args, profile_name=cli_args.profile)
        return

    run_forget(run_args, imagenet=cli_args.imagenet)


if __name__ == "__main__":
    main()
