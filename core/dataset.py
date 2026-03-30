# 1. Standard Library Imports
import json
import math
import os
import random

# 2. Third-Party Imports: Data Science, Math, & Image Processing
import cv2
import numpy as np
import pandas as pd
import skimage as sk
from PIL import Image, ImageFilter, UnidentifiedImageError
from scipy.spatial.transform import Rotation as R

# 3. Third-Party Imports: PyTorch & Torchvision
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

# 4. Local Imports (Relative)
from .data_utils import (
    TransLightning,
    process_viewpoint_label,
    random_crop,
    resize_pad
)


class RotationMNISTDataset(Dataset):
    """
    Wraps torchvision MNIST dataset, applies rotation,
    and outputs (image, labels) where labels = [digit, rotation].
    """

    def __init__(self, base_dataset, transform=None, opt=None,
                 rotation_range=(-90, 90), fixed_rotation=False, num_classes=10):
        """
        Args:
            base_dataset (Dataset): torchvision.datasets.MNIST instance.
            transform: Transform applied after rotation.
            opt: Options object with output_list and output_dimension_list.
            rotation_range (tuple): Min/max angle for random rotations.
            fixed_rotation (bool or int): If int, use fixed rotation angle.
        """
        self.dataset = base_dataset
        self.transform = transform
        self.opt = opt
        self.rotation_range = rotation_range
        self.fixed_rotation = fixed_rotation
        self.num_classes = num_classes

        if opt is not None and hasattr(opt, "output_list"):
            self._validate_and_print_output_config()
        else:
            raise ValueError("opt.output_list must be provided to determine output format")

    def _validate_and_print_output_config(self):
        output_list = self.opt.output_list
        output_dim_list = self.opt.output_dimension_list
        if len(output_list) != len(output_dim_list):
            raise ValueError("output_list and output_dimension_list must have the same length")
        total_dim = sum(output_dim_list)
        print(f"RotationMNIST Dataset Config:")
        print(f"  Output list: {output_list}")
        print(f"  Output dimensions: {output_dim_list}")
        print(f"  Total output dimension: {total_dim}")

    def _get_output_values(self, digit_label, rot_angle):
        output_values = []
        for output_type in self.opt.output_list:
            ot = output_type.lower()
            if "cls" in ot:
                output_values.append(float(digit_label))
            elif "rot" in ot:
                if "sin" in ot:
                    output_values.append(float(torch.sin(torch.tensor(rot_angle * 3.14159 / 180))))
                elif "cos" in ot:
                    output_values.append(float(torch.cos(torch.tensor(rot_angle * 3.14159 / 180))))
                else:
                    output_values.append(float(rot_angle))
            else:
                raise TypeError(f"Unknown output type: {output_type}")
        return output_values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, digit_label = self.dataset[idx]

        # Pick rotation angle
        if isinstance(self.fixed_rotation, int) and self.fixed_rotation:
            rot_angle = self.fixed_rotation
        else:
            rot_angle = random.uniform(*self.rotation_range)

        # Apply rotation
        img = TF.rotate(img, rot_angle)

        # Apply transform (ToTensor, Normalize, etc.)
        img = self.transform(img) if self.transform else TF.to_tensor(img)

        # Outputs
        rot_angle_rad = rot_angle * (torch.pi / 180.0)
        output_values = self._get_output_values(digit_label, rot_angle_rad)
        outputs = torch.tensor(output_values, dtype=torch.float32).unsqueeze(-1)

        return img, outputs

# ImageNet statistics
imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]])
}

# Define normalization and random disturb for input image
disturb = TransLightning(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec'])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def deg2rad(x):
    return x * math.pi / 180.0

class MPIIFaceGazeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the MPIIFaceGaze dataset.

    Assumes the following structure:
      /mnt/data/
          p01/
             p01.txt
             day01/...
          p02/
             p02.txt
             day01/...
          ...

    Each annotation file is expected to have lines with 28 space-separated values:
      1.  Image file path and name (relative to the participant folder)
      2-3. Gaze location (screen coordinate in pixels)
      4-15. Facial landmarks (six (x,y) pairs for four eye corners and two mouth corners)
      16-21. Estimated 3D head pose (rotation and translation based on a 6-point model)
      22-24. Face center in camera coordinate system
      25-27. 3D gaze target location (gaze direction = gaze_target - face_center)
      28.    Which eye is used for evaluation (e.g., "left" or "right")

    In this version, columns 2–27 are converted to floats and the final field is mapped to a numeric
    indicator (0 for "left" and 1 for "right"), yielding a 27-number annotation vector that is split into:
      - gaze_location: indices 0-1 (2 values)
      - facial_landmarks: indices 2-13 (12 values)
      - head_pose: indices 14-19 (6 values)
      - face_center: indices 20-22 (3 values)
      - gaze_target: indices 23-25 (3 values)
      - eye_used: index 26 (1 value)
    """
    def __init__(self, root_dir, transform=None, return_dict=False):
        """
        Args:
            root_dir (str): Path to the root directory containing participant subfolders.
                            Example: "/mnt/data"
            transform (callable, optional): A function/transform to apply to the images.
            return_dict (bool, optional): If True, __getitem__ returns a dict; otherwise a tuple.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_dict = return_dict

        self.samples = []  # Each sample is a dict with keys: "image_path" and "annotation"

        participant = os.path.basename(root_dir)
        annotation_file = f"{root_dir}/{participant}.txt"
        if not os.path.exists(annotation_file):
            print(f"Annotation file {annotation_file} not found, skipping participant {annotation_file}.")

        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # First field is the relative image path (which may include a day folder)
                rel_img_path = parts[0]
                full_img_path = os.path.join(root_dir, rel_img_path)
                # print(rel_img_path)
                # print(full_img_path)
                # Convert columns 2-27 (26 values) to float
                try:
                    annotation_numbers = [float(x) for x in parts[1:27]]
                except ValueError:
                    print(f"Error converting annotation values for {full_img_path}.")
                    continue
                # Process the eye indicator (28th field)
                eye_indicator = parts[27].lower()
                if eye_indicator == "left":
                    eye_val = 0
                elif eye_indicator == "right":
                    eye_val = 1
                else:
                    print(f"Unknown eye indicator '{eye_indicator}' in file {full_img_path}.")
                    continue
                # Create complete annotation vector (27 numbers)
                annotation = annotation_numbers + [eye_val]
                self.samples.append({
                    "image_path": full_img_path,
                    "annotation": annotation
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        # Load image (face region with background blocked as per the dataset release)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        annotation = sample["annotation"]
        # Parse annotation vector into its components
        gaze_location    = torch.tensor(annotation[0:2], dtype=torch.float32)   # 2 numbers
        facial_landmarks = torch.tensor(annotation[2:14], dtype=torch.float32)  # 12 numbers
        head_pose        = torch.tensor(annotation[14:20], dtype=torch.float32)  # 6 numbers
        face_center      = torch.tensor(annotation[20:23], dtype=torch.float32)  # 3 numbers
        gaze_target      = torch.tensor(annotation[23:26], dtype=torch.float32)  # 3 numbers
        eye_used         = int(annotation[26])                                  # 1 number

        if self.return_dict:
            return {
                "image": image,
                "gaze_location": gaze_location,
                "facial_landmarks": facial_landmarks,
                "head_pose": head_pose,
                "face_center": face_center,
                "gaze_target": gaze_target,
                "eye_used": eye_used
            }
        else:
            return image, (gaze_target - face_center).unsqueeze(-1)
        

class MPIIFaceGazeNormDataset(torch.utils.data.Dataset):
    """
    Enhanced MPIIFaceGaze dataset with configurable outputs similar to ShapeNet.
    Supports outputting 3D gaze directions (as az, el, ro) and participant IDs.
    """
    def __init__(self, root_dir, transform=None, return_dict=False, opt=None, participant_to_class=None):
        """
        Args:
            root_dir (str): Path to the root directory containing participant subfolders.
            transform (callable, optional): A function/transform to apply to the images.
            return_dict (bool, optional): If True, __getitem__ returns a dict; otherwise a tuple.
            opt: Options object containing output_list and output_dimension_list
            participant_to_class (dict): Mapping from participant ID to class index
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_dict = return_dict
        self.opt = opt
        self.participant_to_class = participant_to_class or {}
        self.num_classes = len(self.participant_to_class)
        
        # Extract participant ID from root_dir path
        self.participant_id = os.path.basename(root_dir)
        
        # Find the label file in the root directory
        self.label_file = f'{root_dir}.label'
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"Label file not found at {self.label_file}")
            
        # Read the label file
        with open(self.label_file, 'r') as f:
            # Skip header line
            next(f)
            self.data = [line.strip().split() for line in f]
        
        # Validate and print output configuration if opt is provided
        if opt is not None and hasattr(opt, 'output_list'):
            self._validate_and_print_output_config()
    
    def _validate_and_print_output_config(self):
        """Validate output_list configuration and print information."""
        if not hasattr(self.opt, 'output_list') or not hasattr(self.opt, 'output_dimension_list'):
            raise ValueError("opt must have both output_list and output_dimension_list")
            
        output_list = self.opt.output_list
        output_dim_list = self.opt.output_dimension_list
        
        if len(output_list) != len(output_dim_list):
            raise ValueError("output_list and output_dimension_list must have the same length")
        
        total_dim = sum(output_dim_list)
        
        print(f"MPIIFaceGaze Dataset output configuration:")
        print(f"  Output list: {output_list}")
        print(f"  Output dimensions: {output_dim_list}")
        print(f"  Total output dimension: {total_dim}")
        print(f"  Participant ID: {self.participant_id}")
        print(f"  Supported output types: 'id' (participant class), 'x', 'y', 'z' (gaze components)")
        
        # Print expected output order
        output_order = []
        for output_type, dim in zip(output_list, output_dim_list):
            if dim == 1:
                output_order.append(output_type)
            else:
                output_order.extend([f"{output_type}_{i}" for i in range(dim)])
        print(f"  Output order: {output_order}")
    
    def _get_output_values(self, gaze_3d, face_path):
        """Generate output values based on output_list configuration."""
            
        output_list = self.opt.output_list
        
        # Extract participant ID from face_path
        # face_path example: "/path/to/Image/p00/face/001.jpg"
        path_parts = face_path.split(os.sep)
        participant_from_path = None
        for part in path_parts:
            if part.startswith('p') and len(part) == 3:  # e.g., 'p00', 'p01'
                participant_from_path = part
                break
        
        if participant_from_path is None:
            participant_from_path = self.participant_id
        
        output_values = []
        
        for output_type in output_list:
            output_type_lower = output_type.lower()
            
            if 'id' in output_type_lower:
                # Participant ID (only classification type for gaze dataset)
                class_id = self.participant_to_class.get(participant_from_path, 0)
                output_values.append(float(class_id))
                
            elif 'x' in output_type_lower:
                # Direct x component of gaze
                output_values.append(float(gaze_3d.flatten()[0]))
                
            elif 'y' in output_type_lower:
                # Direct y component of gaze
                output_values.append(float(gaze_3d.flatten()[1]))
                
            elif 'z' in output_type_lower:
                # Direct z component of gaze
                output_values.append(float(gaze_3d.flatten()[2]))
                
            else:
                # Unknown output type, raise error
                raise ValueError(f"Output type '{output_type}' not recognized in MPIIFaceGaze dataset configuration. "
                               f"Supported types: 'id', 'x', 'y', 'z'")
        
        return np.array(output_values, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        line = self.data[idx]
        
        # Get face image path and load it
        face_path = os.path.join(os.path.dirname(os.path.dirname(self.root_dir)), 'Image', line[0])
        # print(f"Loading image: {face_path}")  # Debug print
        
        face_image = cv2.imread(face_path)
        if face_image is None:
            raise FileNotFoundError(f"Could not load image at {face_path}")
        
        # Convert BGR to RGB
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = torch.from_numpy(face_image).permute(2, 0, 1).float()  # Shape: (3, H, W)

        # Apply transforms if any
        if self.transform:
            face_image = self.transform(face_image)
            
        # Parse the 3D gaze direction vector (x,y,z)
        gaze_3d = np.array(line[5].split(','), dtype=np.float32).reshape(3, 1)
        gaze_3d = gaze_3d[[2, 0, 1], :]  # Permute indices to order 2,0,1 as in original
        
        # Generate output values based on configuration
        output_values = self._get_output_values(gaze_3d, face_path)
        output_tensor = torch.tensor(output_values, dtype=torch.float32).unsqueeze(-1)
        
        return face_image, output_tensor


class ShapeNetRenderingDataset(Dataset):
    """
    Dataset for ShapeNetRendering: each item is one rendered view and its camera angles.
    Supports classification by adding class labels based on category or individual models.
    Uses unified output_list approach to determine output format.
    """
    def __init__(self, root, samples, transform=None, opt=None, category_to_class=None, 
             model_to_class=None, model_to_id_within_class=None, classification_mode='category', num_classes=None):
        self.root = root
        self.samples = samples
        self.transform = transform
        self.opt = opt  # Store opt for use in __getitem__
        self.category_to_class = category_to_class or {}
        self.model_to_class = model_to_class or {}
        self.model_to_id_within_class = model_to_id_within_class or {}  # NEW: Store within-class ID mapping
        self.classification_mode = classification_mode  # 'category' or 'model'
        self.num_classes = num_classes or 1
        
        # Test augmentation parameters
        self.brightness_severity = 0  # 0 means no augmentation, 1-5 for severity levels
        self.gaussian_noise_severity = 0  # 0 means no augmentation, 1-5 for severity levels
        
        # Validate and print output configuration
        if opt is not None and hasattr(opt, 'output_list'):
            self._validate_and_print_output_config()
        else:
            raise ValueError("opt.output_list must be provided to determine output format")
        
    
    def _validate_and_print_output_config(self):
        """Validate output_list configuration and print information."""
        output_list = self.opt.output_list
        output_dim_list = self.opt.output_dimension_list
        
        if len(output_list) != len(output_dim_list):
            raise ValueError("output_list and output_dimension_list must have the same length")
        
        total_dim = sum(output_dim_list)
        
        print(f"Dataset output configuration:")
        print(f"  Output list: {output_list}")
        print(f"  Output dimensions: {output_dim_list}")
        print(f"  Total output dimension: {total_dim}")
        
        # Check for classification outputs
        classification_keywords = self.opt.classification_keywords
        has_classification = any(any(keyword in output_type.lower() for keyword in classification_keywords) 
                               for output_type in output_list)
        # Print expected output order
        output_order = []
        for output_type, dim in zip(output_list, output_dim_list):
            if dim == 1:
                output_order.append(output_type)
            else:
                output_order.extend([f"{output_type}_{i}" for i in range(dim)])
        print(f"  Output order: {output_order}")
    
    def _get_output_values(self, az, el, ip, category, obj_id):
        """Generate output values based on output_list configuration."""
        output_list = self.opt.output_list
        output_dim_list = getattr(self.opt, 'output_dimension_list', [1] * len(output_list))
        
        # Pre-compute trigonometric values
        sin_az = np.sin(az)
        cos_az = np.cos(az)
        sin_el = np.sin(el)
        cos_el = np.cos(el)
        sin_ip = np.sin(ip)
        cos_ip = np.cos(ip)
        
        output_values = []
        
        for output_type, dim in zip(output_list, output_dim_list):
            output_type_lower = output_type.lower()
            
            # Handle different classification types
            if 'cls' in output_type_lower:
                # Category-based classification (respects current classification_mode)
                if self.classification_mode == 'category':
                    class_label = self.category_to_class.get(category, 0)
                else:
                    model_id = f"{category}/{obj_id}"
                    class_label = self.model_to_class.get(model_id, 0)
                output_values.append(float(class_label))
                
            elif 'id' in output_type_lower:
                # Model ID within class (0-indexed within each class)
                model_id = f"{category}/{obj_id}"
                id_within_class = getattr(self, 'model_to_id_within_class', {}).get(model_id, 0)
                output_values.append(float(id_within_class))
                    
            elif 'az' in output_type_lower:
                # Azimuth-related outputs
                if 'sin' in output_type_lower:
                    output_values.append(sin_az)
                elif 'cos' in output_type_lower:
                    output_values.append(cos_az)
                else:
                    # Raw azimuth angle
                    output_values.append(az)
                    
            elif 'el' in output_type_lower:
                # Elevation-related outputs
                if 'sin' in output_type_lower:
                    output_values.append(sin_el)
                elif 'cos' in output_type_lower:
                    output_values.append(cos_el)
                else:
                    # Raw elevation angle
                    output_values.append(el)
                    
            elif 'ip' in output_type_lower or 'ro' in output_type_lower:
                # In-plane rotation related outputs
                if 'sin' in output_type_lower:
                    output_values.append(sin_ip)
                elif 'cos' in output_type_lower:    
                    output_values.append(cos_ip)
                else:
                    # Raw in-plane rotation angle
                    output_values.append(ip)
            else:
                # Unknown output type, default to zero
                raise TypeError(f"output type '{output_type}' not recognized in the dataset configuration")
        
        return output_values
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load sample
        category, obj_id, view_idx = self.samples[idx]
        base = os.path.join(self.root, category, obj_id, 'rendering')
        img_path = os.path.join(base, f"{view_idx:03d}.png")
        
        # if not os.path.exists(img_path):
        #     # Path not exists, try other path
        #     img_path = os.path.join(base, f"{view_idx:02d}.png")
        #     if not os.path.exists(img_path):
        #         # Path doesn't exist
        #         print(f"Image not found: {img_path}")
        
        # Load image safely
        img = Image.open(img_path).convert('RGB')
        
        # Convert to numpy array for augmentations
        img_array = np.array(img)
        
        # Convert back to PIL Image
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply transform
        img = self.transform(img) if self.transform else TF.to_tensor(img)
        
        # Load metadata
        meta_path = os.path.join(base, 'rendered_images_metadata.txt')
        lines = []
        if os.path.isfile(meta_path):
            with open(meta_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
        
        if view_idx >= len(lines):
            raise ValueError(f"Metadata missing for {category}/{obj_id} view {view_idx}")
        
        parts = lines[view_idx].split()
        az = float(parts[0])
        el = float(parts[1])
        ip = float(parts[2])
        d = float(parts[3])
        
        # Convert to radians
        az = math.radians(az)
        el = math.radians(el)
        ip = math.radians(ip)
        
        # Generate output values based on output_list configuration
        output_values = self._get_output_values(az, el, ip, category, obj_id)
        
        cam_params = torch.tensor(output_values, dtype=torch.float32)
        cam_params = cam_params.unsqueeze(-1)

        return img, cam_params

class ShapeNetRenderingDataset_Retrieval(Dataset):
    """
    Dataset for ShapeNetRendering retrieval task: each item returns two images from the same model
    with their corresponding camera parameters.
    """
    def __init__(self, root, transform=None, opt=None, category_to_class=None, num_classes=None):
        self.root = root
        self.transform = transform
        self.opt = opt
        self.category_to_class = category_to_class or {}
        self.num_classes = num_classes or 1
        
        # Discover all models in the dataset
        self.models = self._discover_models()
        
        # Test augmentation parameters
        self.brightness_severity = 0
        self.gaussian_noise_severity = 0
        
        # Print output format info
        if opt is not None and "az_el_ip_no_split" == opt.output_type:
            print("Retrieval dataset - the order of output is : az, el, ip")
        elif opt is not None and "az_el_ip" == opt.output_type:
            print("Retrieval dataset - the order of output is : sin(az), cos(az), sin(el), cos(el), sin(ip), cos(ip)")
        elif opt is not None and "az_el_no_split" == opt.output_type:
            print("Retrieval dataset - the order of output is : az, el")
        elif opt is not None and "az_el_az_el_no_split" == opt.output_type:
            print("Retrieval dataset - the order of output is : az, el, az, el")
        elif opt is not None and "az_el_az_el_az_el_no_split" == opt.output_type:
            print("Retrieval dataset - the order of output is : az, el, az, el, az, el")
        elif opt is not None and "az_el" == opt.output_type:
            print("Retrieval dataset - the order of output is : sin(az), cos(az), sin(el), cos(el)")
        elif opt is not None and "az" == opt.output_type:
            print("Retrieval dataset - the order of output is : sin(az), cos(az)")
        else:
            raise ValueError("provide output format in trial")
        
        print(f"Retrieval dataset loaded with {len(self.models)} models")
        
        # Print classification info if enabled
        if opt is not None and hasattr(opt, 'classification') and opt.classification:
            print(f"Retrieval classification enabled with {self.num_classes} classes")
            print(f"Category to class mapping: {self.category_to_class}")
    
    def _discover_models(self):
        """Discover all models in the dataset directory structure."""
        models = []
        
        # Iterate through category directories
        for category in os.listdir(self.root):
            category_path = os.path.join(self.root, category)
            if not os.path.isdir(category_path):
                continue
            
            # Iterate through model directories
            for model_id in os.listdir(category_path):
                model_path = os.path.join(category_path, model_id)
                if not os.path.isdir(model_path):
                    continue
                
                # Check if model has exactly 2 images
                rendering_path = os.path.join(model_path, 'rendering')
                if not os.path.isdir(rendering_path):
                    continue
                
                # Look for image files (both .png formats)
                image_files = []
                for file in os.listdir(rendering_path):
                    if file.endswith('.png'):
                        image_files.append(file)
                
                # Sort to ensure consistent ordering
                image_files.sort()
                
                if len(image_files) == 2:
                    # Extract view indices from filenames
                    view_indices = []
                    for img_file in image_files:
                        try:
                            # Try 3-digit format first (001.png)
                            if len(img_file) == 7:  # "001.png"
                                view_idx = int(img_file[:3])
                            # Try 2-digit format (01.png)
                            elif len(img_file) == 6:  # "01.png"
                                view_idx = int(img_file[:2])
                            else:
                                continue
                            view_indices.append(view_idx)
                        except ValueError:
                            continue
                    
                    if len(view_indices) == 2:
                        models.append((category, model_id, view_indices[0], view_indices[1]))
                        
        print(f"Found {len(models)} models with exactly 2 images each")
        return models
    
    def set_augmentation_severity(self, brightness_severity=0, gaussian_noise_severity=0):
        """Set augmentation severity levels for testing robustness."""
        self.brightness_severity = brightness_severity
        self.gaussian_noise_severity = gaussian_noise_severity
    
    def brightness(self, x, severity=1):
        """Apply brightness augmentation."""
        c = [.1, .2, .3, .4, .5][severity - 1]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.clip(x, 0, 1) * 255
    
    def gaussian_noise(self, x, severity=1):
        """Apply gaussian noise augmentation."""
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    
    def apply_augmentations(self, img_array):
        """Apply all enabled augmentations to the image."""
        # Apply brightness augmentation
        if self.brightness_severity > 0:
            img_array = self.brightness(img_array, self.brightness_severity)
        
        # Apply gaussian noise augmentation
        if self.gaussian_noise_severity > 0:
            img_array = self.gaussian_noise(img_array, self.gaussian_noise_severity)
        
        return img_array
    
    def _load_camera_params(self, category, model_id, view_idx):
        """Load camera parameters for a specific view."""
        base = os.path.join(self.root, category, model_id, 'rendering')
        meta_path = os.path.join(base, 'rendered_images_metadata.txt')
        
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        if view_idx >= len(lines):
            raise ValueError(f"Metadata missing for {category}/{model_id} view {view_idx}")
        
        parts = lines[view_idx].split()
        az = float(parts[0])
        el = float(parts[1])
        ip = float(parts[2])
        d = float(parts[3])
        
        az = math.radians(az)
        el = math.radians(el)
        ip = math.radians(ip)
        
        # Compute sin/cos values
        sin_az = np.sin(az)
        cos_az = np.cos(az)
        sin_el = np.sin(el)
        cos_el = np.cos(el)
        sin_ip = np.sin(ip)
        cos_ip = np.cos(ip)
        
        # Create parameter tensor based on opt.output_type
        if self.opt is not None and "az_el_ip_no_split" == self.opt.output_type:
            cam_params = torch.tensor([az, el, ip], dtype=torch.float32)
        elif self.opt is not None and "az_el_ip" == self.opt.output_type:
            cam_params = torch.tensor([sin_az, cos_az, sin_el, cos_el, sin_ip, cos_ip], dtype=torch.float32)
        elif self.opt is not None and "az_el_no_split" == self.opt.output_type:
            cam_params = torch.tensor([az, el], dtype=torch.float32)
        elif self.opt is not None and "az_el_az_el_no_split" == self.opt.output_type:
            cam_params = torch.tensor([az, el, az, el], dtype=torch.float32)
        elif self.opt is not None and "az_el_az_el_az_el_no_split" == self.opt.output_type:
            cam_params = torch.tensor([az, el, az, el, az, el], dtype=torch.float32)
        elif self.opt is not None and "az_el" == self.opt.output_type:
            cam_params = torch.tensor([sin_az, cos_az, sin_el, cos_el], dtype=torch.float32)
        elif self.opt is not None and "az" == self.opt.output_type:
            cam_params = torch.tensor([sin_az, cos_az], dtype=torch.float32)
        else:
            raise ValueError("provide output format in trial")
        
        cam_params = cam_params.unsqueeze(-1)
        
        # Add classification label if enabled
        if self.opt.classification:
            class_label = self.category_to_class.get(category, 0)
            class_label = torch.tensor([class_label], dtype=torch.float32)
            cam_params = torch.cat([cam_params, class_label.unsqueeze(-1)], dim=0)
        
        return cam_params
    
    def _load_image(self, category, model_id, view_idx):
        """Load and process a single image."""
        base = os.path.join(self.root, category, model_id, 'rendering')
        img_path = os.path.join(base, f"{view_idx:03d}.png")
        
        if not os.path.exists(img_path):
            # Try 2-digit format
            img_path = os.path.join(base, f"{view_idx:02d}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Convert to numpy array for augmentations
        img_array = np.array(img)
        
        # Apply augmentations BEFORE transform
        img_array = self.apply_augmentations(img_array)
        
        # Convert back to PIL Image
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply transform
        img = self.transform(img) if self.transform else TF.to_tensor(img)
        
        return img
    
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, idx):
        category, model_id, view_idx1, view_idx2 = self.models[idx]
        
        # Load both images
        image1 = self._load_image(category, model_id, view_idx1)
        image2 = self._load_image(category, model_id, view_idx2)
        
        # Load camera parameters for both views
        cam_param1 = self._load_camera_params(category, model_id, view_idx1)
        cam_param2 = self._load_camera_params(category, model_id, view_idx2)
        
        return image1, image2, cam_param1, cam_param2
    
