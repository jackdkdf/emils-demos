"""Entry point for the ML project.

Provides a tiny CLI to run common actions (placeholder).
"""
import argparse
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser("your-ml-project")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("about", help="Show project info")

    args = parser.parse_args(argv)

    if args.command == "about":
        print("your-ml-project: starter project layout. See README.md for details.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
