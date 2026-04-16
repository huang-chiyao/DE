#!/usr/bin/env python3
import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_utils import (
    DEFAULT_RENDER_ROOT,
    RENDERED_IMG_METADATA,
    RENDERING_SUBFOLDER,
    count_rendered_images,
    discover_rendered_models,
    get_rendering_dir,
)

def detect_parameter_generation_type(metadata, param_name='az'):
    """
    Detect if parameters were generated using discrete or continuous sampling.
    Args:
        metadata: list of [az, el, ro, d_ratio, fov] values
        param_name: 'az' or 'el' to analyze
    Returns:
        'discrete' or 'continuous'
    """
    if not metadata:
        return 'unknown'
    
    # Extract the parameter values
    param_idx = {'az': 0, 'el': 1, 'ro': 2, 'd_ratio': 3, 'fov': 4}
    if param_name not in param_idx:
        return 'unknown'
    
    values = np.array([row[param_idx[param_name]] for row in metadata])
    
    # Check for discrete patterns
    unique_values = np.unique(values)
    
    # Method 1: Check if values follow discrete step pattern
    if len(unique_values) <= 10:  # Small number of unique values suggests discrete
        # Check if differences between consecutive unique values are consistent
        if len(unique_values) > 1:
            diffs = np.diff(sorted(unique_values))
            # If most differences are the same (within tolerance), it's likely discrete
            if len(np.unique(np.round(diffs))) <= 2:  # Allow for some floating point errors
                return 'discrete'
    
    # Method 2: Check if values are "too regular" for continuous
    # Continuous should have more varied decimal places
    decimal_places = []
    for val in values:
        # Count decimal places
        str_val = f"{val:.6f}".rstrip('0').rstrip('.')
        if '.' in str_val:
            decimal_places.append(len(str_val.split('.')[1]))
        else:
            decimal_places.append(0)
    
    # If most values have 0 decimal places or very few, likely discrete
    avg_decimal_places = np.mean(decimal_places)
    if avg_decimal_places < 0.5:  # Most values are integers
        return 'discrete'
    
    # Method 3: Check distribution uniformity
    # Discrete values often cluster at specific intervals
    if len(unique_values) < len(values) * 0.3:  # Less than 30% unique values
        return 'discrete'
    
    return 'continuous'

def analyze_model_parameters(category, model_id, target_images=10, render_root=None):
    """
    Analyze a model's parameters to determine generation type.
    Returns:
        dict with analysis results
    """
    rendering_dir = get_rendering_dir(category, model_id, render_root)
    metadata_file = rendering_dir / RENDERED_IMG_METADATA
    
    # Count images
    if not rendering_dir.exists():
        return {'valid': False, 'complete': False, 'img_count': 0, 'label_count': 0}
    
    try:
        img_count = count_rendered_images(category, model_id, render_root)
    except Exception as e:
        img_count = 0
    
    # Load metadata
    metadata = []
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # az, el, ro, d_ratio, fov
                        az, el, ro, d_ratio, fov = map(float, parts[:5])
                        metadata.append([az, el, ro, d_ratio, fov])
        except Exception as e:
            print(f"Error reading metadata for {category}/{model_id}: {e}")
    
    label_count = len(metadata)
    
    # Model validity checks
    is_valid = img_count == label_count and img_count > 0
    is_complete = img_count == target_images and label_count == target_images
    
    # Only analyze parameters if model is valid
    az_type = el_type = 'unknown'
    if is_valid and metadata:
        az_type = detect_parameter_generation_type(metadata, 'az')
        el_type = detect_parameter_generation_type(metadata, 'el')
    
    return {
        'valid': is_valid,
        'complete': is_complete,
        'img_count': img_count,
        'label_count': label_count,
        'az_type': az_type,
        'el_type': el_type,
        'metadata': metadata
    }

def categorize_models_by_category(all_models, target_images=10, render_root=None):
    """
    Categorize models by category and then by parameter generation type.
    Returns:
        dict with category as key and (discrete_models, continuous_models, invalid_models) as values
    """
    # Group models by category
    models_by_category = defaultdict(list)
    for category, model_id in all_models:
        models_by_category[category].append(model_id)
    
    results = {}
    
    print("Analyzing parameter generation types by category...")
    print("=" * 80)
    
    for category in sorted(models_by_category.keys()):
        print(f"\n📂 CATEGORY: {category}")
        print("-" * 60)
        
        models = models_by_category[category]
        discrete_models = []
        continuous_models = []
        invalid_models = []
        
        for idx, model_id in enumerate(models):
            analysis = analyze_model_parameters(category, model_id, target_images, render_root)
            
            if not analysis['valid']:
                invalid_models.append((idx, category, model_id, analysis['img_count'], analysis['label_count']))
                status = f"❌ INVALID ({analysis['img_count']} imgs, {analysis['label_count']} labels)"
            else:
                # Determine overall type (both az and el should be consistent)
                if analysis['az_type'] == 'discrete' and analysis['el_type'] == 'discrete':
                    discrete_models.append((category, model_id, analysis))
                    status = f"🎯 DISCRETE (az: {analysis['az_type']}, el: {analysis['el_type']})"
                elif analysis['az_type'] == 'continuous' or analysis['el_type'] == 'continuous':
                    continuous_models.append((category, model_id, analysis))
                    status = f"🌊 CONTINUOUS (az: {analysis['az_type']}, el: {analysis['el_type']})"
                else:
                    # Mixed or unknown - put in continuous to be safe
                    continuous_models.append((category, model_id, analysis))
                    status = f"❓ MIXED/UNKNOWN (az: {analysis['az_type']}, el: {analysis['el_type']})"
            
            print(f"  Model {idx:3d}: {model_id} - {status}")
        
        # Summary for this category
        print(f"  📊 {category} Summary:")
        print(f"    🎯 Discrete: {len(discrete_models)}")
        print(f"    🌊 Continuous: {len(continuous_models)}")
        print(f"    ❌ Invalid: {len(invalid_models)}")
        
        results[category] = (discrete_models, continuous_models, invalid_models)
    
    return results

def generate_view_samples(category, model_id, max_views, render_root=None):
    """Generate view-level samples for a valid model."""
    rendering_dir = get_rendering_dir(category, model_id, render_root)
    
    if not rendering_dir.exists():
        return []
    
    try:
        image_files = sorted([f for f in os.listdir(rendering_dir) if f.endswith('.png')])
        samples = []
        for i, img_file in enumerate(image_files[:max_views]):
            view_idx = int(os.path.splitext(img_file)[0])  # assumes file is '000.png', etc.
            samples.append((category, model_id, view_idx))
        return samples
    except:
        return []

def distribute_samples_across_models(
    valid_models,
    target_images,
    train_ratio,
    val_ratio,
    shuffle,
    split_strategy='view_level',
    render_root=None,
):
    """
    NEW FUNCTION: Distribute samples more evenly across models.
    
    Args:
        valid_models: List of (category, model_id) tuples
        target_images: Images per model
        train_ratio: Ratio for training
        val_ratio: Ratio for validation (of train+val)
        shuffle: Whether to shuffle
        split_strategy: 'view_level' or 'model_level' or 'balanced_model'
    
    Returns:
        (train_samples, val_samples, test_samples)
    """
    
    if split_strategy == 'view_level':
        # Strategy 1: Collect ALL views from ALL models, then split at view level
        print(f"🔄 Using VIEW-LEVEL splitting strategy")
        all_view_samples = []
        for category, model_id in valid_models:
            view_samples = generate_view_samples(category, model_id, target_images, render_root)
            all_view_samples.extend(view_samples)
        
        if shuffle:
            random.shuffle(all_view_samples)
        
        total_views = len(all_view_samples)
        train_val_split = int(total_views * train_ratio)
        val_count = max(1, int(train_val_split * val_ratio)) if train_val_split > 1 else 0
        
        train_samples = all_view_samples[:train_val_split - val_count]
        val_samples = all_view_samples[train_val_split - val_count:train_val_split]
        test_samples = all_view_samples[train_val_split:]
        
        return train_samples, val_samples, test_samples
    
    elif split_strategy == 'balanced_model':
        # Strategy 2: Split views from EACH model proportionally
        print(f"🔄 Using BALANCED-MODEL splitting strategy")
        train_samples = []
        val_samples = []
        test_samples = []
        
        for category, model_id in valid_models:
            view_samples = generate_view_samples(category, model_id, target_images, render_root)
            
            if shuffle:
                random.shuffle(view_samples)
            
            num_views = len(view_samples)
            if num_views == 0:
                continue
                
            # Calculate splits for this model
            train_val_count = max(1, int(num_views * train_ratio))
            val_count = max(1, int(train_val_count * val_ratio)) if train_val_count > 1 else 0
            train_count = train_val_count - val_count
            
            # Ensure we don't exceed available views
            if train_count + val_count > num_views:
                train_count = max(1, num_views - val_count)
                val_count = min(val_count, num_views - train_count)
            
            # Split this model's views
            train_samples.extend(view_samples[:train_count])
            val_samples.extend(view_samples[train_count:train_count + val_count])
            test_samples.extend(view_samples[train_count + val_count:])
        
        return train_samples, val_samples, test_samples
    
    else:  # 'model_level' - original behavior
        print(f"🔄 Using MODEL-LEVEL splitting strategy (original)")
        if shuffle:
            random.shuffle(valid_models)
        
        # Split models first
        split_idx = int(len(valid_models) * train_ratio)
        train_val_models = valid_models[:split_idx]
        test_models = valid_models[split_idx:]
        
        # Further split train_val into train and val
        if shuffle:
            random.shuffle(train_val_models)
        val_count = max(1, int(len(train_val_models) * val_ratio)) if len(train_val_models) > 1 else 0
        val_models = train_val_models[:val_count]
        train_models = train_val_models[val_count:]
        
        # Generate view samples
        train_samples = []
        val_samples = []
        test_samples = []
        
        for category, model_id in train_models:
            train_samples.extend(generate_view_samples(category, model_id, target_images, render_root))
        
        for category, model_id in val_models:
            val_samples.extend(generate_view_samples(category, model_id, target_images, render_root))
            
        for category, model_id in test_models:
            test_samples.extend(generate_view_samples(category, model_id, target_images, render_root))
        
        return train_samples, val_samples, test_samples

def process_category_splits(category, discrete_models, continuous_models, invalid_models, 
                          target_images, train_ratio, val_ratio, shuffle, base_cache_dir,
                          max_models_per_category=None, use_discrete_continuous_split=True,
                          split_strategy='view_level', render_root=None):
    """
    Process splits for a single category and save to category-specific directory.
    
    Args:
        max_models_per_category: Maximum number of models to use per category (None = no limit)
        use_discrete_continuous_split: If True, use discrete/continuous logic. If False, simple train/test split
        split_strategy: 'view_level', 'model_level', or 'balanced_model'
    """
    # Create category-specific cache directory
    category_cache_dir = os.path.join(base_cache_dir, category)
    os.makedirs(category_cache_dir, exist_ok=True)
    
    print(f"\n📂 PROCESSING CATEGORY: {category}")
    print(f"💾 Output directory: {category_cache_dir}")
    print(f"🔧 Mode: {'Discrete/Continuous Split' if use_discrete_continuous_split else 'Simple Train/Test Split'}")
    print(f"🎯 Split strategy: {split_strategy}")
    if max_models_per_category:
        print(f"📊 Model limit: {max_models_per_category} models per category")
    print("-" * 60)
    
    # Initialize samples
    train_samples = []
    val_samples = []
    test_samples = []
    test_continuous_samples = []
    
    # Combine all valid models if using simple split mode
    if not use_discrete_continuous_split:
        all_valid_models = []
        all_valid_models.extend([(cat, mid) for cat, mid, _ in discrete_models])
        all_valid_models.extend([(cat, mid) for cat, mid, _ in continuous_models])
        
        # Apply model limit if specified
        if max_models_per_category and len(all_valid_models) > max_models_per_category:
            if shuffle:
                random.shuffle(all_valid_models)
            all_valid_models = all_valid_models[:max_models_per_category]
            print(f"🔢 Limited to {len(all_valid_models)} models (from {len(discrete_models) + len(continuous_models)} total)")
        
        # Use the new distribution strategy
        train_samples, val_samples, test_samples = distribute_samples_across_models(
            all_valid_models, target_images, train_ratio, val_ratio, shuffle, split_strategy, render_root
        )
        
        print(f"🎯 SIMPLE SPLIT RESULTS:")
        print(f"  Train: {len(train_samples)} view samples")
        print(f"  Val: {len(val_samples)} view samples") 
        print(f"  Test: {len(test_samples)} view samples")
        
        # No continuous test samples in simple mode
        test_continuous_samples = []
        
    else:
        # Original discrete/continuous logic with model limit
        discrete_model_pairs = [(cat, mid) for cat, mid, _ in discrete_models]
        continuous_model_pairs = [(cat, mid) for cat, mid, _ in continuous_models]
        
        # Apply model limit to discrete models if specified
        if max_models_per_category and len(discrete_model_pairs) > max_models_per_category:
            if shuffle:
                random.shuffle(discrete_model_pairs)
            discrete_model_pairs = discrete_model_pairs[:max_models_per_category]
            print(f"🔢 Limited discrete models to {len(discrete_model_pairs)} (from {len(discrete_models)} total)")
        
        # Apply model limit to continuous models if specified  
        if max_models_per_category and len(continuous_model_pairs) > max_models_per_category:
            if shuffle:
                random.shuffle(continuous_model_pairs)
            continuous_model_pairs = continuous_model_pairs[:max_models_per_category]
            print(f"🔢 Limited continuous models to {len(continuous_model_pairs)} (from {len(continuous_models)} total)")
    
        # ===== PROCESS DISCRETE MODELS =====
        if len(discrete_model_pairs) > 0:
            print("🎯 PROCESSING DISCRETE MODELS:")
            print("-" * 40)
            
            # Use the new distribution strategy for discrete models
            train_samples, val_samples, test_samples = distribute_samples_across_models(
                discrete_model_pairs, target_images, train_ratio, val_ratio, shuffle, split_strategy, render_root
            )
            
            print(f"Discrete model results:")
            print(f"  Train: {len(train_samples)} view samples")
            print(f"  Val: {len(val_samples)} view samples")
            print(f"  Test: {len(test_samples)} view samples")
        
        else:
            print("🎯 NO DISCRETE MODELS FOUND")
        
        # ===== PROCESS CONTINUOUS MODELS =====
        if len(continuous_model_pairs) > 0:
            print(f"\n🌊 PROCESSING CONTINUOUS MODELS:")
            print("-" * 40)
            
            # All continuous models go to test_continuous
            for category, model_id in continuous_model_pairs:
                view_samples = generate_view_samples(category, model_id, target_images, render_root)
                test_continuous_samples.extend(view_samples)
            
            print(f"Continuous models: {len(continuous_model_pairs)} models")
            print(f"Test continuous: {len(test_continuous_samples)} view samples")
        
        else:
            print("🌊 NO CONTINUOUS MODELS FOUND")
    
    # ===== WRITE OUTPUT FILES =====
    print(f"\n📝 WRITING OUTPUT FILES:")
    print("-" * 40)
    
    def write_samples(samples, filename):
        if len(samples) == 0:
            print(f"  ⚠️ Skipping {filename} - no samples")
            return
        
        filepath = os.path.join(category_cache_dir, filename)
        with open(filepath, 'w') as f:
            for category, model_id, view_idx in samples:
                f.write(f"{category}/{model_id} {view_idx:03d}\n")
        print(f"  ✅ {filename}: {len(samples)} samples")
    
    # Write all splits
    write_samples(train_samples, 'train.txt')
    write_samples(val_samples, 'val.txt')
    write_samples(test_samples, 'test.txt')
    write_samples(test_continuous_samples, 'test_continuous.txt')
    
    return {
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'test_continuous_samples': len(test_continuous_samples),
        'discrete_models': len(discrete_models),
        'continuous_models': len(continuous_models),
        'invalid_models': len(invalid_models)
    }

def _default_render_root():
    return DEFAULT_RENDER_ROOT


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits from rendered ShapeNet outputs."
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=_default_render_root(),
        help="Root directory that contains rendered categories/models.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where split txt files are written. Defaults to --render-root.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category ids to process. Defaults to all discovered categories.",
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=300,
        help="Expected max number of rendered images per model.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction reserved for train+val before test split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train+val reserved for validation.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["view_level", "model_level", "balanced_model"],
        default="view_level",
        help="How to distribute samples across train/val/test.",
    )
    parser.add_argument(
        "--max-models-per-category",
        type=int,
        default=None,
        help="Optional cap on models used per category.",
    )
    parser.add_argument(
        "--use-discrete-continuous-split",
        action="store_true",
        help="Enable the original discrete-vs-continuous split behavior.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when shuffling.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before splitting.",
    )
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = _parse_cli_args()

    target_images = args.target_images
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    shuffle = not args.no_shuffle
    render_root = Path(args.render_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else render_root
    max_models_per_category = args.max_models_per_category
    use_discrete_continuous_split = args.use_discrete_continuous_split
    split_strategy = args.split_strategy

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["SHAPENET_RENDER_ROOT"] = str(render_root)
    
    print("Smart Train/Test Splitter with Improved Model Distribution")
    print("=" * 80)
    print(f"Rendering root: {render_root}")
    print(f"Output root: {output_root}")
    print(f"Target images per model: {target_images}")
    print(f"Train ratio: {train_ratio}, Val ratio: {val_ratio}")
    print(f"Max models per category: {max_models_per_category if max_models_per_category else 'No limit'}")
    print(f"Split mode: {'Discrete/Continuous' if use_discrete_continuous_split else 'Simple Train/Test'}")
    print(f"Split strategy: {split_strategy}")
    print("-" * 80)
    
    # Explain split strategies
    strategy_explanations = {
        'view_level': "All views from all models pooled together, then split randomly",
        'model_level': "Models split first, then all views from each model go to respective split",
        'balanced_model': "Each model contributes proportionally to train/val/test splits"
    }
    print(f"📋 Split strategy explanation:")
    print(f"   {split_strategy}: {strategy_explanations.get(split_strategy, 'Unknown strategy')}")
    print("-" * 80)
    
    all_models = discover_rendered_models(render_root, args.categories)
    
    print(f"Total models to analyze: {len(all_models)}")
    if not all_models:
        print("No rendered models found. Nothing to split.")
        return 1
    
    # Categorize models by category and parameter generation type
    results_by_category = categorize_models_by_category(all_models, target_images, render_root)
    
    # Process each category separately
    print("\n" + "=" * 80)
    print("PROCESSING CATEGORY SPLITS")
    print("=" * 80)
    
    overall_stats = {
        'total_categories': len(results_by_category),
        'total_train_samples': 0,
        'total_val_samples': 0,
        'total_test_samples': 0,
        'total_test_continuous_samples': 0,
        'total_discrete_models': 0,
        'total_continuous_models': 0,
        'total_invalid_models': 0
    }
    
    for category in sorted(results_by_category.keys()):
        discrete_models, continuous_models, invalid_models = results_by_category[category]
        
        category_stats = process_category_splits(
            category, discrete_models, continuous_models, invalid_models,
            target_images, train_ratio, val_ratio, shuffle, str(output_root),
            max_models_per_category, use_discrete_continuous_split, split_strategy, render_root
        )
        
        # Update overall stats
        overall_stats['total_train_samples'] += category_stats['train_samples']
        overall_stats['total_val_samples'] += category_stats['val_samples']
        overall_stats['total_test_samples'] += category_stats['test_samples']
        overall_stats['total_test_continuous_samples'] += category_stats['test_continuous_samples']
        overall_stats['total_discrete_models'] += category_stats['discrete_models']
        overall_stats['total_continuous_models'] += category_stats['continuous_models']
        overall_stats['total_invalid_models'] += category_stats['invalid_models']
    
    # ===== FINAL SUMMARY =====
    print(f"\n" + "=" * 80)
    print("✅ SPLIT GENERATION COMPLETED!")
    print("=" * 80)
    print(f"📁 Base output directory: {output_root}")
    print(f"📂 Categories processed: {overall_stats['total_categories']}")
    print(f"📋 Overall statistics:")
    print(f"  🎯 Total train samples: {overall_stats['total_train_samples']}")
    print(f"  🎯 Total val samples: {overall_stats['total_val_samples']}")
    print(f"  🎯 Total test samples: {overall_stats['total_test_samples']}")
    print(f"  🌊 Total test continuous samples: {overall_stats['total_test_continuous_samples']}")
    print(f"  📊 Total discrete models: {overall_stats['total_discrete_models']}")
    print(f"  📊 Total continuous models: {overall_stats['total_continuous_models']}")
    print(f"  ❌ Total invalid models: {overall_stats['total_invalid_models']}")
    
    total_samples = (overall_stats['total_train_samples'] + 
                    overall_stats['total_val_samples'] + 
                    overall_stats['total_test_samples'] + 
                    overall_stats['total_test_continuous_samples'])
    print(f"  📊 Grand total samples: {total_samples}")
    
    print(f"\n🔍 PARAMETER GENERATION ANALYSIS:")
    print(f"  Models using discrete generation (random.choice): {overall_stats['total_discrete_models']}")
    print(f"  Models using continuous generation (random.uniform): {overall_stats['total_continuous_models']}")
    print(f"  Invalid models (excluded): {overall_stats['total_invalid_models']}")
    
    print(f"\n📂 DIRECTORY STRUCTURE:")
    print(f"  {output_root}/")
    for category in sorted(results_by_category.keys()):
        print(f"    ├── {category}/")
        print(f"    │   ├── train.txt")
        print(f"    │   ├── val.txt")
        print(f"    │   ├── test.txt")
        if use_discrete_continuous_split:
            print(f"    │   └── test_continuous.txt")
    
    print(f"\n🔧 CONFIGURATION NOTES:")
    print(f"  • Split mode: {'Discrete/Continuous' if use_discrete_continuous_split else 'Simple Train/Test'}")
    print(f"  • Split strategy: {split_strategy}")
    print(f"  • Model limit per category: {max_models_per_category if max_models_per_category else 'No limit'}")
    print(f"  • Each category has its own split files")
    print(f"  • To change split strategy: use --split-strategy")
    print(f"      - 'view_level': Best distribution across models")
    print(f"      - 'balanced_model': Each model contributes to all splits")
    print(f"      - 'model_level': Original behavior (separate model sets)")
    print(f"  • To change split mode: pass --use-discrete-continuous-split")
    print(f"  • To change model limit: pass --max-models-per-category")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
    if USE_DISCRETE_CONTINUOUS_SPLIT:
        print(f"  • Detection assumes discrete uses random.choice() with step intervals")
        print(f"  • Detection assumes continuous uses random.uniform() with float values")
    else:
        print(f"  • Simple mode: combines all valid models for train/test split")

if __name__ == '__main__':
    main()
