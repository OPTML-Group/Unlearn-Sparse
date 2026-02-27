import arg_parser
from pruning_pipeline import run_pruning


def main():
    args = arg_parser.parse_args()
    run_pruning(args, profile_name="ls")


if __name__ == "__main__":
    main()
