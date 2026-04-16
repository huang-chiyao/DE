#!/usr/bin/env python3

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_utils import (
    count_rendered_images,
    discover_rendered_models,
    get_rendering_dir,
    load_render_metadata,
)

def detect_parameter_type(values, param_name):
    """
    Detect if parameter values are discrete or continuous and their expected range.
    
    Args:
        values: numpy array of parameter values
        param_name: name of the parameter ('az', 'el', 'ro', 'd_ratio', 'fov')
    
    Returns:
        dict with 'type' ('discrete' or 'continuous'), 'range', and 'unique_count'
    """
    unique_values = np.unique(values)
    unique_count = len(unique_values)
    total_count = len(values)
    
    # Determine expected ranges based on parameter name
    expected_ranges = {
        'az': [(0, 359), (-180, 180)],  # Common ranges for azimuth
        'el': [(0, 90), (-45, 45)],     # Common ranges for elevation
        'ro': [(-180, 179), (-180, 180)],  # Rotation ranges
        'd_ratio': [(1, 3), (0.5, 5)],     # Distance ratio ranges
        'fov': [(59, 60), (30, 90)]        # FOV ranges
    }
    
    # Detect if discrete (small number of unique values relative to total)
    # or if values appear to be evenly spaced
    if unique_count <= 20 or unique_count / total_count < 0.1:
        param_type = 'discrete'
    else:
        # Check if values are roughly evenly distributed (continuous)
        if unique_count > total_count * 0.8:
            param_type = 'continuous'
        else:
            param_type = 'discrete'
    
    # Determine which range the data fits
    min_val, max_val = values.min(), values.max()
    detected_range = None
    
    if param_name in expected_ranges:
        for range_option in expected_ranges[param_name]:
            range_min, range_max = range_option
            # Check if data fits within this range (with some tolerance)
            if min_val >= range_min - 5 and max_val <= range_max + 5:
                detected_range = range_option
                break
    
    if detected_range is None:
        detected_range = (min_val, max_val)
    
    return {
        'type': param_type,
        'range': detected_range,
        'unique_count': unique_count,
        'actual_range': (min_val, max_val)
    }

def analyze_parameter_distribution(values, param_name):
    """Analyze parameter distribution and return detailed info."""
    info = detect_parameter_type(values, param_name)
    
    # Additional statistics
    info.update({
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75)
    })
    
    return info

def load_model_data(category, model_id):
    """Load both image list and metadata for a single model."""
    rendering_dir = get_rendering_dir(category, model_id)
    
    # Count images
    # Directly count PNG files instead of relying on an img list file
    if not rendering_dir.exists():
        return 0, []
    
    try:
        img_count = count_rendered_images(category, model_id)
    except Exception as e:
        print(f"Error accessing images for {category}/{model_id}: {e}")
        img_count = 0

    # Load metadata
    try:
        metadata = load_render_metadata(category, model_id)
    except Exception as e:
        print(f"Error reading metadata for {category}/{model_id}: {e}")
        metadata = []
    
    return img_count, metadata

def create_enhanced_plots(param_data, param_names):
    """Create enhanced plots showing both discrete and continuous distributions."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Enhanced Parameter Distribution Analysis', fontsize=16)
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (param_name, values) in enumerate(zip(param_names, param_data)):
        if len(values) == 0:
            continue
            
        info = analyze_parameter_distribution(values, param_name)
        row, col = i // 2, i % 2
        
        if i >= 5:  # Only handle first 5 parameters
            break
            
        ax = axes[row, col]
        
        # Create histogram
        if info['type'] == 'discrete':
            # For discrete data, use specific bins for each unique value
            unique_vals = np.unique(values)
            if len(unique_vals) <= 50:  # Reasonable number of discrete values
                bins = np.concatenate([unique_vals - 0.5, [unique_vals[-1] + 0.5]])
                ax.hist(values, bins=bins, alpha=0.7, color=colors[i], edgecolor='black')
            else:
                ax.hist(values, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        else:
            # For continuous data, use regular binning
            ax.hist(values, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        
        # Add statistics
        ax.axvline(info['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {info["mean"]:.1f}')
        ax.axvline(info['median'], color='darkred', linestyle=':', linewidth=2, label=f'Median: {info["median"]:.1f}')
        
        # Title with distribution type and range info
        title = f'{param_name.upper()} - {info["type"].title()}\n'
        title += f'Range: {info["range"]}, Unique: {info["unique_count"]}'
        ax.set_title(title, fontsize=10)
        
        ax.set_xlabel(f'{param_name} values')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Special plot for azimuth vs elevation (last subplot)
    if len(param_data) >= 2:
        ax = axes[2, 1]
        az_values, el_values = param_data[0], param_data[1]
        
        if len(az_values) > 0 and len(el_values) > 0:
            # Create scatter plot with density information
            ax.scatter(az_values, el_values, alpha=0.5, s=1, c='navy')
            ax.set_title('Azimuth vs Elevation Distribution')
            ax.set_xlabel('Azimuth (degrees)')
            ax.set_ylabel('Elevation (degrees)')
            ax.grid(True, alpha=0.3)
            
            # Add range indicators
            az_info = analyze_parameter_distribution(az_values, 'az')
            el_info = analyze_parameter_distribution(el_values, 'el')
            
            ax.text(0.05, 0.95, f'Az: {az_info["type"]}, Range: {az_info["range"]}\n'
                              f'El: {el_info["type"]}, Range: {el_info["range"]}', 
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    NUM_MODELS_PER_CATEGORY = 10
    TARGET_IMAGES = 300
    
    # Get all rendered models directly from the output tree
    all_models = discover_rendered_models()
    
    print(f"Total models found: {len(all_models)}")
    print(f"Analyzing {NUM_MODELS_PER_CATEGORY} models per category")
    print(f"Expected: {TARGET_IMAGES} images per model")
    print("=" * 80)
    
    # Group all models by category first
    all_category_models = defaultdict(list)
    for category, model_id in all_models:
        all_category_models[category].append(model_id)
    
    # Select NUM_MODELS_PER_CATEGORY from each category
    selected_models = []
    category_models = {}
    global_idx = 0
    
    print(f"CATEGORY AVAILABILITY:")
    print("-" * 50)
    
    for category, models in all_category_models.items():
        available_models = len(models)
        models_to_take = min(NUM_MODELS_PER_CATEGORY, available_models)
        
        print(f"  {category}: {available_models} available, taking {models_to_take}")
        
        # Take the first models_to_take models from this category
        category_selected = []
        for i in range(models_to_take):
            model_id = models[i]
            selected_models.append((category, model_id))
            category_selected.append((global_idx, model_id))
            global_idx += 1
        
        category_models[category] = category_selected
    
    total_selected = len(selected_models)
    print(f"\nTotal models selected: {total_selected}")
    print(f"Expected total: {len(all_category_models) * NUM_MODELS_PER_CATEGORY}")
    
    # Print category breakdown
    print(f"\nCATEGORY BREAKDOWN ({NUM_MODELS_PER_CATEGORY} models per category):")
    print("-" * 50)
    for category, models in category_models.items():
        print(f"  {category}: {len(models)} models")
    
    # Initialize overall tracking
    overall_stats = {
        'models_with_data': 0,
        'total_images_found': 0,
        'total_labels_found': 0,
        'perfect_models': 0,
        'mismatched_models': []
    }
    
    # Store data for each category
    category_data = {}
    
    print(f"\n" + "=" * 80)
    print("DETAILED ANALYSIS BY CATEGORY:")
    print("=" * 80)
    
    # Process each category
    for category, models in category_models.items():
        print(f"\n📁 CATEGORY: {category}")
        print(f"   Models to analyze: {len(models)}")
        print("-" * 60)
        
        # Initialize category-specific data
        cat_az = []
        cat_el = []
        cat_ro = []
        cat_d_ratio = []
        cat_fov = []
        
        cat_stats = {
            'models_with_data': 0,
            'total_images_found': 0,
            'total_labels_found': 0,
            'perfect_models': 0,
            'mismatched_models': []
        }
        
        # Process each model in this category
        for idx, model_id in models:
            img_count, metadata = load_model_data(category, model_id)
            label_count = len(metadata)
            
            # Check if counts match
            is_mismatch = img_count != label_count
            is_complete = img_count == TARGET_IMAGES and label_count == TARGET_IMAGES
            
            if img_count > 0 or label_count > 0:
                cat_stats['models_with_data'] += 1
                overall_stats['models_with_data'] += 1
            
            cat_stats['total_images_found'] += img_count
            cat_stats['total_labels_found'] += label_count
            overall_stats['total_images_found'] += img_count
            overall_stats['total_labels_found'] += label_count
            
            if is_complete:
                cat_stats['perfect_models'] += 1
                overall_stats['perfect_models'] += 1
            
            if is_mismatch:
                cat_stats['mismatched_models'].append((idx, category, model_id, img_count, label_count))
                overall_stats['mismatched_models'].append((idx, category, model_id, img_count, label_count))
            
            # Status indicator
            if is_mismatch:
                status = "⚠️  MISMATCH"
            elif is_complete:
                status = "✅ PERFECT"
            elif img_count == 0 and label_count == 0:
                status = "❌ NO DATA"
            else:
                status = f"🔄 PARTIAL ({min(img_count, label_count)}/{TARGET_IMAGES})"
            
            print(f"   Model {idx:3d}: {model_id} - Images: {img_count:3d}, Labels: {label_count:3d} - {status}")
            
            # Only add data if counts match (to avoid misaligned data)
            if not is_mismatch and metadata:
                for az, el, ro, d_ratio, fov in metadata:
                    cat_az.append(az)
                    cat_el.append(el)
                    cat_ro.append(ro)
                    cat_d_ratio.append(d_ratio)
                    cat_fov.append(fov)
        
        # Category summary
        expected_total = len(models) * TARGET_IMAGES
        print(f"\n   📊 CATEGORY {category} SUMMARY:")
        print(f"      Models with data: {cat_stats['models_with_data']}/{len(models)}")
        print(f"      Perfect models: {cat_stats['perfect_models']}")
        print(f"      Models with mismatches: {len(cat_stats['mismatched_models'])}")
        print(f"      Total images found: {cat_stats['total_images_found']}")
        print(f"      Total labels found: {cat_stats['total_labels_found']}")
        print(f"      Expected total: {expected_total}")
        print(f"      Image coverage: {cat_stats['total_images_found']/expected_total*100:.1f}%")
        print(f"      Label coverage: {cat_stats['total_labels_found']/expected_total*100:.1f}%")
        print(f"      Data points for analysis: {len(cat_az)}")
        
        # Store category data
        category_data[category] = {
            'param_data': [np.array(cat_az), np.array(cat_el), np.array(cat_ro), 
                          np.array(cat_d_ratio), np.array(cat_fov)],
            'param_names': ['az', 'el', 'ro', 'd_ratio', 'fov'],
            'stats': cat_stats,
            'model_count': len(models)
        }
    
    # Overall summary
    print(f"\n" + "=" * 80)
    print("OVERALL SUMMARY:")
    print("=" * 80)
    expected_total = total_selected * TARGET_IMAGES
    print(f"Total models analyzed: {total_selected}")
    print(f"Models with data: {overall_stats['models_with_data']}/{total_selected}")
    print(f"Perfect models: {overall_stats['perfect_models']}")
    print(f"Models with mismatches: {len(overall_stats['mismatched_models'])}")
    print(f"Total images found: {overall_stats['total_images_found']}")
    print(f"Total labels found: {overall_stats['total_labels_found']}")
    print(f"Expected total: {expected_total}")
    print(f"Image coverage: {overall_stats['total_images_found']/expected_total*100:.1f}%")
    print(f"Label coverage: {overall_stats['total_labels_found']/expected_total*100:.1f}%")
    
    # Create plots for each category
    print(f"\n" + "=" * 80)
    print("PARAMETER ANALYSIS BY CATEGORY:")
    print("=" * 80)
    
    for category, data in category_data.items():
        param_data = data['param_data']
        param_names = data['param_names']
        
        # Skip if no data
        if len(param_data[0]) == 0:
            print(f"\n❌ No data for category {category} - skipping analysis")
            continue
        
        print(f"\n🔍 CATEGORY: {category}")
        print(f"   Data points: {len(param_data[0])}")
        print("-" * 50)
        
        # Parameter analysis for this category
        for param_name, values in zip(param_names, param_data):
            if len(values) == 0:
                continue
                
            info = analyze_parameter_distribution(values, param_name)
            
            print(f"   {param_name.upper()} - {info['type'].upper()}:")
            print(f"     Range: {info['range']} (actual: {info['actual_range'][0]:.1f} to {info['actual_range'][1]:.1f})")
            print(f"     Stats: mean={info['mean']:.2f}, std={info['std']:.2f}")
            print(f"     Unique values: {info['unique_count']}/{len(values)} ({info['unique_count']/len(values)*100:.1f}%)")
        
        # Create plot for this category
        if len(param_data[0]) > 0:
            fig = create_enhanced_plots(param_data, param_names)
            fig.suptitle(f'Parameter Distribution Analysis - Category: {category}', fontsize=16)
            
            plot_filename = f'parameter_distributions_{category}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"   📊 Plot saved as '{plot_filename}'")
    
    # Create combined plot with all categories
    print(f"\n" + "=" * 80)
    print("CREATING COMBINED ANALYSIS:")
    print("=" * 80)
    
    # Combine all data
    all_az = []
    all_el = []
    all_ro = []
    all_d_ratio = []
    all_fov = []
    
    for category, data in category_data.items():
        param_data = data['param_data']
        all_az.extend(param_data[0])
        all_el.extend(param_data[1])
        all_ro.extend(param_data[2])
        all_d_ratio.extend(param_data[3])
        all_fov.extend(param_data[4])
    
    if all_az:
        # Convert to numpy arrays
        all_az = np.array(all_az)
        all_el = np.array(all_el)
        all_ro = np.array(all_ro)
        all_d_ratio = np.array(all_d_ratio)
        all_fov = np.array(all_fov)
        
        # Combined analysis
        combined_param_data = [all_az, all_el, all_ro, all_d_ratio, all_fov]
        param_names = ['az', 'el', 'ro', 'd_ratio', 'fov']
        
        print(f"\n🔍 COMBINED ANALYSIS (all categories):")
        print(f"   Total data points: {len(all_az)}")
        print("-" * 50)
        
        for param_name, values in zip(param_names, combined_param_data):
            info = analyze_parameter_distribution(values, param_name)
            
            print(f"   {param_name.upper()} - {info['type'].upper()}:")
            print(f"     Range: {info['range']} (actual: {info['actual_range'][0]:.1f} to {info['actual_range'][1]:.1f})")
            print(f"     Stats: mean={info['mean']:.2f}, std={info['std']:.2f}")
            print(f"     Unique values: {info['unique_count']}/{len(values)} ({info['unique_count']/len(values)*100:.1f}%)")
        
        # Create combined plot
        fig = create_enhanced_plots(combined_param_data, param_names)
        fig.suptitle('Combined Parameter Distribution Analysis - All Categories', fontsize=16)
        plt.savefig('parameter_distributions_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Combined plot saved as 'parameter_distributions_combined.png'")
        print(f"   📝 Analysis complete with {len(all_az)} total data points")
    else:
        print(f"\n❌ No valid data found across all categories!")
    
    # Final category comparison
    print(f"\n" + "=" * 80)
    print("CATEGORY COMPARISON:")
    print("=" * 80)
    
    print(f"{'Category':<15} {'Models':<8} {'Perfect':<8} {'Data Points':<12} {'Coverage':<10}")
    print("-" * 65)
    
    for category, data in category_data.items():
        model_count = data['model_count']
        perfect_count = data['stats']['perfect_models']
        data_points = len(data['param_data'][0])
        coverage = data['stats']['total_images_found'] / (model_count * TARGET_IMAGES) * 100
        
        print(f"{category:<15} {model_count:<8} {perfect_count:<8} {data_points:<12} {coverage:<10.1f}%")

if __name__ == '__main__':
    main()
