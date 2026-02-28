import argparse

from src.core import arg_parser


def build_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Unified entrypoint for all workflows.\n\n"
            "Entries:\n"
            "  train    Train + iterative pruning (replaces main_imp.py/main_ls.py/main_sam.py/main_vit.py/main_synflow.py)\n"
            "  prune    Legacy alias of 'train'\n"
            "  unlearn  Run machine unlearning (replaces main_forget.py/main_forget_imagenet.py)\n"
            "  backdoor Run backdoor cleanse experiment (replaces main_backdoor.py)"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_train_profile_arg(subparser):
        subparser.add_argument(
        "--profile",
        default="imp",
        choices=["imp", "ls", "sam", "vit", "synflow"],
        help=(
            "Training profile (with pruning strategy):\n"
            "  imp -> standard IMP/OMP training pipeline\n"
            "  ls  -> label-smoothing variant\n"
            "  sam -> SAM optimizer variant\n"
            "  vit -> ViT-style optimizer/scheduler variant\n"
            "  synflow -> SynFlow pruning pipeline"
        ),
    )

    train = subparsers.add_parser("train", help="Train + iterative pruning")
    _add_train_profile_arg(train)

    # Keep backward compatibility with old command name.
    prune = subparsers.add_parser("prune", help="Legacy alias of 'train'")
    _add_train_profile_arg(prune)

    unlearn = subparsers.add_parser("unlearn", help="Run unlearning")
    unlearn.add_argument(
        "--imagenet",
        action="store_true",
        help=(
            "Force ImageNet unlearning pipeline.\n"
            "Usually not needed: --dataset imagenet is auto-detected."
        ),
    )

    subparsers.add_parser(
        "backdoor", help="Run backdoor cleanse pipeline (legacy: main_backdoor.py)"
    )
    return parser


def main():
    cli = build_cli()
    cli_args, passthrough = cli.parse_known_args()
    run_args = arg_parser.parse_args(passthrough)

    if cli_args.command in {"train", "prune"}:
        from src.pipelines.train_pipeline import run_pruning, run_synflow

        if cli_args.profile == "synflow":
            run_synflow(run_args)
        else:
            run_pruning(run_args, profile_name=cli_args.profile)
        return

    if cli_args.command == "backdoor":
        from src.pipelines.backdoor_pipeline import run_backdoor

        run_backdoor(run_args)
        return

    from src.pipelines.forget_pipeline import run_forget

    inferred_imagenet = getattr(run_args, "dataset", None) == "imagenet"
    run_forget(run_args, imagenet=(cli_args.imagenet or inferred_imagenet))


if __name__ == "__main__":
    main()
