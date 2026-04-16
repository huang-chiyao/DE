#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_utils import count_rendered_images, discover_rendered_models

def count_model_images(category, model_id):
    """Count rendered images for a single model."""
    return count_rendered_images(category, model_id)

def main():
    TARGET_IMAGES = 600  # Expected number of images per model
    
    # Get rendered models directly from the rendering output tree
    all_models = discover_rendered_models()[0:100]
    
    print(f"Checking {len(all_models)} models...")
    print(f"Target: {TARGET_IMAGES} images per model")
    print("-" * 60)
    
    completed_models = 0
    incomplete_models = 0
    total_images = 0
    
    incomplete_list = []
    
    for idx, (category, model_id) in enumerate(all_models):
        img_count = count_model_images(category, model_id)
        total_images += img_count
        
        status = "COMPLETE" if img_count == TARGET_IMAGES else f"INCOMPLETE ({img_count}/{TARGET_IMAGES})"
        
        if img_count == TARGET_IMAGES:
            completed_models += 1
        else:
            incomplete_models += 1
            incomplete_list.append((idx, category, model_id, img_count))
        
        print(f"Model {idx:3d}: {category}/{model_id} - {img_count:3d} images - {status}")
    
    print("-" * 60)
    print(f"SUMMARY:")
    print(f"  Total models: {len(all_models)}")
    print(f"  Completed: {completed_models}")
    print(f"  Incomplete: {incomplete_models}")
    print(f"  Total images rendered: {total_images}")
    print(f"  Expected total: {len(all_models) * TARGET_IMAGES}")
    print(f"  Progress: {completed_models/len(all_models)*100:.1f}%")
    
    if incomplete_list:
        print(f"\nINCOMPLETE MODELS:")
        for idx, category, model_id, count in incomplete_list:
            print(f"  Model {idx}: {category}/{model_id} - {count} images")
    
    print(f"\nTo resume rendering, set START_IDX to the first incomplete model index.")

if __name__ == '__main__':
    main()
