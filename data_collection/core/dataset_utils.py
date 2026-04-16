#!/usr/bin/env python3
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "ShapeNet" / "ShapeNetRendering_az_90_el_ro_45_cls_id_30k"
RENDERING_SUBFOLDER = "rendering"
RENDERED_IMG_LIST = "rendered_images.txt"
RENDERED_IMG_METADATA = "rendered_images_metadata.txt"


def resolve_render_root(render_root=None):
    return Path(render_root).resolve() if render_root else DEFAULT_RENDER_ROOT.resolve()


def get_rendering_dir(category, model_id, render_root=None):
    root = resolve_render_root(render_root)
    return root / category / model_id / RENDERING_SUBFOLDER


def discover_categories(render_root=None):
    root = resolve_render_root(render_root)
    if not root.exists():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir()])


def discover_rendered_models(render_root=None, categories=None):
    root = resolve_render_root(render_root)
    requested_categories = set(categories) if categories else None
    models = []

    if not root.exists():
        return models

    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if requested_categories and category_dir.name not in requested_categories:
            continue

        for model_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            rendering_dir = model_dir / RENDERING_SUBFOLDER
            if rendering_dir.exists():
                models.append((category_dir.name, model_dir.name))

    return models


def count_rendered_images(category, model_id, render_root=None):
    rendering_dir = get_rendering_dir(category, model_id, render_root)
    if not rendering_dir.exists():
        return 0
    return len([name for name in os.listdir(rendering_dir) if name.endswith(".png")])


def load_render_metadata(category, model_id, render_root=None):
    metadata_path = get_rendering_dir(category, model_id, render_root) / RENDERED_IMG_METADATA
    metadata = []

    if not metadata_path.exists():
        return metadata

    with open(metadata_path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 5:
                metadata.append(list(map(float, parts[:5])))

    return metadata
