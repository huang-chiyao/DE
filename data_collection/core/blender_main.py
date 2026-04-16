#!/usr/bin/env python3
import argparse
import os
import sys


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Blender entrypoint: render multiple ShapeNet categories."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="ShapeNet category ids, e.g. --categories 02691156 02958343",
    )
    parser.add_argument(
        "--begin_idx",
        type=int,
        required=True,
        help="Starting model index (inclusive)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        required=True,
        help="Ending model index (exclusive)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images per model (default: 10)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--render_root",
        default=None,
        help="Optional output directory for rendered images.",
    )
    parser.add_argument(
        "--shapenet_root",
        default=None,
        help="Optional ShapeNetVox32 root directory.",
    )

    # When run through Blender, Blender passes extra args; filter after '--'
    if "blender" in sys.argv[0].lower():
        try:
            script_args_start = sys.argv.index("--") + 1
            args_to_parse = sys.argv[script_args_start:]
        except ValueError:
            args_to_parse = []
        return parser.parse_args(args_to_parse)

    return parser.parse_args()


def main():
    # Ensure we can import render_models_3.py from the same directory.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    from render_models_3 import main as render_models_main

    cli = _parse_cli_args()
    categories = cli.categories

    for category in categories:
        print(f"\n=== Rendering category {category} ===")
        render_models_main(
            category,
            cli.begin_idx,
            cli.end_idx,
            cli.num_images,
            cli.num_processes,
            cli.render_root,
            cli.shapenet_root,
        )


if __name__ == "__main__":
    main()
