#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_utils import DEFAULT_RENDER_ROOT, discover_categories, load_render_metadata

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

def load_model_metadata(category, model_id):
    """Load metadata for a single model."""
    try:
        return load_render_metadata(category, model_id)
    except Exception as e:
        print(f"Error reading metadata for {category}/{model_id}: {e}")
        return []

def load_split_data(split_file_path):
    """Load split data from a txt file."""
    data = []
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    model_path, view_idx_str = parts
                    category, model_id = model_path.split('/')
                    view_idx = int(view_idx_str)
                    data.append((category, model_id, view_idx))
    return data

def create_enhanced_plots(param_data, param_names, title_prefix):
    """Create enhanced plots showing both discrete and continuous distributions."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'{title_prefix} - Parameter Distribution Analysis', fontsize=16)
    
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
    BASE_DIR = str(DEFAULT_RENDER_ROOT)
    
    # Get all categories
    categories = discover_categories(BASE_DIR)
    
    splits = ['train', 'val', 'test']
    param_names = ['az', 'el', 'ro', 'd_ratio', 'fov']
    
    # Initialize data collection
    split_data = {}
    for split in splits:
        split_data[split] = {param: [] for param in param_names}
    
    # Cache metadata to avoid reloading
    metadata_cache = {}
    
    print("Loading split data and metadata...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Categories found: {len(categories)}")
    print("=" * 80)
    
    total_entries = 0
    
    for category in categories:
        category_dir = os.path.join(BASE_DIR, category)
        print(f"\nProcessing category: {category}")
        
        category_entries = 0
        
        for split in splits:
            split_file = os.path.join(category_dir, f'{split}.txt')
            entries = load_split_data(split_file)
            
            print(f"  {split}.txt: {len(entries)} entries")
            category_entries += len(entries)
            
            for cat, model_id, view_idx in entries:
                key = (cat, model_id)
                if key not in metadata_cache:
                    metadata_cache[key] = load_model_metadata(cat, model_id)
                
                metadata = metadata_cache[key]
                if view_idx < len(metadata):
                    az, el, ro, d_ratio, fov = metadata[view_idx]
                    split_data[split]['az'].append(az)
                    split_data[split]['el'].append(el)
                    split_data[split]['ro'].append(ro)
                    split_data[split]['d_ratio'].append(d_ratio)
                    split_data[split]['fov'].append(fov)
        
        total_entries += category_entries
        print(f"  Total entries for {category}: {category_entries}")
    
    print(f"\nTotal entries across all splits: {total_entries}")
    print(f"Metadata cache size: {len(metadata_cache)} models")
    
    # Convert to numpy arrays
    for split in splits:
        for param in param_names:
            split_data[split][param] = np.array(split_data[split][param])
    
    # Print statistics and create plots for each split
    print(f"\n" + "=" * 80)
    print("PARAMETER STATISTICS BY SPLIT:")
    print("=" * 80)
    
    for split in splits:
        param_data = [split_data[split][param] for param in param_names]
        num_samples = len(param_data[0]) if len(param_data) > 0 else 0
        
        print(f"\n🔍 SPLIT: {split.upper()}")
        print(f"   Samples: {num_samples}")
        print("-" * 50)
        
        for param_name, values in zip(param_names, param_data):
            if len(values) == 0:
                continue
                
            info = analyze_parameter_distribution(values, param_name)
            
            print(f"   {param_name.upper()} - {info['type'].upper()}:")
            print(f"     Range: {info['range']} (actual: {info['actual_range'][0]:.1f} to {info['actual_range'][1]:.1f})")
            print(f"     Stats: mean={info['mean']:.2f}, std={info['std']:.2f}, median={info['median']:.2f}")
            print(f"     Quartiles: Q25={info['q25']:.2f}, Q75={info['q75']:.2f}")
            print(f"     Unique values: {info['unique_count']}/{len(values)} ({info['unique_count']/len(values)*100:.1f}%)")
        
        # Create plot for this split
        if num_samples > 0:
            fig = create_enhanced_plots(param_data, param_names, f'Split: {split.upper()}')
            plot_filename = f'parameter_distributions_{split}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()  # Free memory
            
            print(f"   📊 Plot saved as '{plot_filename}'")
    
    # Create combined plot with all splits
    print(f"\n" + "=" * 80)
    print("CREATING COMBINED ANALYSIS:")
    print("=" * 80)
    
    # Combine all data
    all_az = []
    all_el = []
    all_ro = []
    all_d_ratio = []
    all_fov = []
    
    for split in splits:
        all_az.extend(split_data[split]['az'])
        all_el.extend(split_data[split]['el'])
        all_ro.extend(split_data[split]['ro'])
        all_d_ratio.extend(split_data[split]['d_ratio'])
        all_fov.extend(split_data[split]['fov'])
    
    if all_az:
        # Convert to numpy arrays
        all_az = np.array(all_az)
        all_el = np.array(all_el)
        all_ro = np.array(all_ro)
        all_d_ratio = np.array(all_d_ratio)
        all_fov = np.array(all_fov)
        
        # Combined analysis
        combined_param_data = [all_az, all_el, all_ro, all_d_ratio, all_fov]
        
        print(f"\n🔍 COMBINED ANALYSIS (all splits):")
        print(f"   Total data points: {len(all_az)}")
        print("-" * 50)
        
        for param_name, values in zip(param_names, combined_param_data):
            info = analyze_parameter_distribution(values, param_name)
            
            print(f"   {param_name.upper()} - {info['type'].upper()}:")
            print(f"     Range: {info['range']} (actual: {info['actual_range'][0]:.1f} to {info['actual_range'][1]:.1f})")
            print(f"     Stats: mean={info['mean']:.2f}, std={info['std']:.2f}, median={info['median']:.2f}")
            print(f"     Quartiles: Q25={info['q25']:.2f}, Q75={info['q75']:.2f}")
            print(f"     Unique values: {info['unique_count']}/{len(values)} ({info['unique_count']/len(values)*100:.1f}%)")
        
        # Create combined plot
        fig = create_enhanced_plots(combined_param_data, param_names, 'Combined - All Splits')
        plt.savefig('parameter_distributions_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Combined plot saved as 'parameter_distributions_combined.png'")
        print(f"   📝 Analysis complete with {len(all_az)} total data points")
    else:
        print(f"\n❌ No valid data found across all splits!")
    
    # Final split comparison
    print(f"\n" + "=" * 80)
    print("SPLIT COMPARISON:")
    print("=" * 80)
    
    print(f"{'Split':<8} {'Samples':<10} {'Az Type':<10} {'El Type':<10} {'Ro Type':<10}")
    print("-" * 60)
    
    for split in splits:
        param_data = [split_data[split][param] for param in param_names]
        num_samples = len(param_data[0]) if len(param_data) > 0 else 0
        
        if num_samples > 0:
            az_info = analyze_parameter_distribution(split_data[split]['az'], 'az')
            el_info = analyze_parameter_distribution(split_data[split]['el'], 'el')
            ro_info = analyze_parameter_distribution(split_data[split]['ro'], 'ro')
            
            print(f"{split:<8} {num_samples:<10} {az_info['type']:<10} {el_info['type']:<10} {ro_info['type']:<10}")
        else:
            print(f"{split:<8} {num_samples:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

if __name__ == '__main__':
    main()
