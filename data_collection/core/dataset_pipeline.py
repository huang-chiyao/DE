#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _project_root():
    return Path(__file__).resolve().parent.parent


def _default_render_root():
    return _project_root() / "ShapeNet" / "ShapeNetRendering_az_90_el_ro_45_cls_id_30k"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for rendering and split generation."
    )
    parser.add_argument(
        "--mode",
        choices=["render", "split", "all"],
        default="all",
        help="Which stages to run.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="ShapeNet category ids to process.",
    )
    parser.add_argument(
        "--begin-idx",
        type=int,
        default=0,
        help="Starting model index for rendering.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Ending model index (exclusive) for rendering.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=300,
        help="Number of rendered images per model.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Parallel worker count for rendering.",
    )
    parser.add_argument(
        "--blender-bin",
        default="blender",
        help="Blender executable.",
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=_default_render_root(),
        help="Rendered dataset root. Also used as render output when mode includes render.",
    )
    parser.add_argument(
        "--shapenet-root",
        type=Path,
        default=None,
        help="Optional ShapeNetVox32 root used for rendering input meshes.",
    )
    parser.add_argument(
        "--split-output-root",
        type=Path,
        default=None,
        help="Where split txt files should be written. Defaults to --render-root.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["view_level", "model_level", "balanced_model"],
        default="view_level",
        help="Split strategy passed to core/split.py.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction reserved for train+val.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train+val reserved for validation.",
    )
    parser.add_argument(
        "--max-models-per-category",
        type=int,
        default=None,
        help="Optional model cap per category during splitting.",
    )
    parser.add_argument(
        "--use-discrete-continuous-split",
        action="store_true",
        help="Enable original discrete/continuous split handling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by split generation.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling during split generation.",
    )
    return parser.parse_args()


def _run_command(cmd, cwd):
    print("\n$ " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_render(args, project_root):
    if args.end_idx is None:
        raise ValueError("--end-idx is required when mode includes render")

    cmd = [
        args.blender_bin,
        "-b",
        "-P",
        str(project_root / "core" / "blender_main.py"),
        "--",
        "--categories",
        *args.categories,
        "--begin_idx",
        str(args.begin_idx),
        "--end_idx",
        str(args.end_idx),
        "--num_images",
        str(args.num_images),
        "--num_processes",
        str(args.num_processes),
        "--render_root",
        str(args.render_root),
    ]
    if args.shapenet_root is not None:
        cmd.extend(["--shapenet_root", str(args.shapenet_root)])
    _run_command(cmd, project_root)


def _run_split(args, project_root):
    split_output_root = args.split_output_root or args.render_root
    cmd = [
        sys.executable,
        str(project_root / "core" / "split.py"),
        "--render-root",
        str(args.render_root),
        "--output-root",
        str(split_output_root),
        "--categories",
        *args.categories,
        "--target-images",
        str(args.num_images),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--split-strategy",
        args.split_strategy,
        "--seed",
        str(args.seed),
    ]

    if args.max_models_per_category is not None:
        cmd.extend(["--max-models-per-category", str(args.max_models_per_category)])
    if args.use_discrete_continuous_split:
        cmd.append("--use-discrete-continuous-split")
    if args.no_shuffle:
        cmd.append("--no-shuffle")

    _run_command(cmd, project_root)


def main():
    args = _parse_args()
    project_root = _project_root()

    if args.mode in {"render", "all"}:
        _run_render(args, project_root)
    if args.mode in {"split", "all"}:
        _run_split(args, project_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
