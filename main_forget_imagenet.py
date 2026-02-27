import arg_parser
from forget_pipeline import run_forget


def main():
    args = arg_parser.parse_args()
    run_forget(args, imagenet=True)


if __name__ == "__main__":
    main()
