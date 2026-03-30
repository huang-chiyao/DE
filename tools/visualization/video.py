import numpy as np
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
import cv2
import glob
import time

def parse_option():
    parser = argparse.ArgumentParser('plotting')
    parser.add_argument('--data_prefix', type=str, default='00', help='dataset prefix')
    parser.add_argument('--output_dir', type=str, default='/scratch/chuan166/vocal/latent/', help='output directory for combined images')
    opt = parser.parse_args()
    return opt

opt = parse_option()

# Create output directory (with data_prefix subfolder)
data_dir = os.path.join(opt.output_dir, opt.data_prefix)
os.makedirs(data_dir, exist_ok=True)

# Load the results from the NPZ file (for latent space and trajectory)
results = np.load(f'results/results_{opt.data_prefix}.npz')

# Define DOF names (for latent space plots)
dof_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

# Precompute PCA results and labels for each DOF.
pca_dict = {}
labels_dict = {}
for dof in dof_names:
    features = results[f'feature_{dof}']
    labels = results[f'label_{dof}']
    # Standardize features
    mean = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    std_dev[std_dev == 0] = 1
    standardized_features = (features - mean) / std_dev

    # PCA transformation (3 components)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(standardized_features)
    pca_dict[dof] = pca_result
    labels_dict[dof] = labels

# ---------------- Trajectory Computation ----------------
# Prepare 6-DoF arrays for predictions and labels.
n_samples = len(results['label_x'])
predicts = np.zeros((n_samples, 6))
predicts[:, 0] = results['output_x'].flatten()
predicts[:, 1] = results['output_y'].flatten()
predicts[:, 2] = results['output_z'].flatten()
predicts[:, 3] = results['output_roll'].flatten()
predicts[:, 4] = results['output_pitch'].flatten()
predicts[:, 5] = results['output_yaw'].flatten()

labels_arr = np.zeros((n_samples, 6))
labels_arr[:, 0] = results['label_x'].flatten()
labels_arr[:, 1] = results['label_y'].flatten()
labels_arr[:, 2] = results['label_z'].flatten()
labels_arr[:, 3] = results['label_roll'].flatten()
labels_arr[:, 4] = results['label_pitch'].flatten()
labels_arr[:, 5] = results['label_yaw'].flatten()

# Accumulate transformation matrices for ground truth
labels_mat = np.zeros((n_samples, 4, 4))
new_labels = np.zeros((n_samples, 4, 4))
current_p = np.eye(4)
for i in range(n_samples):
    rl = R.from_rotvec(labels_arr[i, 3:6])
    labels_mat[i, 0:3, 0:3] = rl.as_matrix()
    labels_mat[i, 0:3, 3] = labels_arr[i, 0:3]
    labels_mat[i, 3, 3] = 1.0
    current_p = np.matmul(current_p, labels_mat[i])
    new_labels[i] = current_p

# Accumulate transformation matrices for predictions
predicts_mat = np.zeros((n_samples, 4, 4))
new_predicts = np.zeros((n_samples, 4, 4))
current_p = np.eye(4)
for i in range(n_samples):
    rp = R.from_rotvec(predicts[i, 3:6])
    predicts_mat[i, 0:3, 0:3] = rp.as_matrix()
    predicts_mat[i, 0:3, 3] = predicts[i, 0:3]
    predicts_mat[i, 3, 3] = 1.0
    current_p = np.matmul(current_p, predicts_mat[i])
    new_predicts[i] = current_p

# Extract 2D trajectory coordinates (using x and z).
xl = new_labels[:, 0, 3]
zl = new_labels[:, 2, 3]
xp = new_predicts[:, 0, 3]
zp = new_predicts[:, 2, 3]
# ---------------- End Trajectory Computation ----------------

# Assume all latent features have the same number of frames.
n_frames = next(iter(pca_dict.values())).shape[0]

# For each frame, create a 2×4 figure.
for frame_idx in range(n_frames):
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4)

    # --- Latent Space Subplots (3D) for row 0, columns 0-2 (x, y, z) ---
    latent_positions = {
        'x': (0, 0),
        'y': (0, 1),
        'z': (0, 2),
        'roll': (1, 0),
        'pitch': (1, 1),
        'yaw': (1, 2)
    }
    for dof in ['x', 'y', 'z']:
        pos = latent_positions[dof]
        ax = fig.add_subplot(gs[pos[0], pos[1]], projection='3d')
        pca_result = pca_dict[dof]
        lab = labels_dict[dof]
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                   c=lab, cmap='viridis', alpha=0.9)
        current_point = pca_result[frame_idx]
        ax.text(current_point[0], current_point[1], current_point[2],
                '○', color='red', fontsize=24, ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=3, foreground='red')])
        ax.set_title(f"KITTI {opt.data_prefix} " + r"$\mathbf{" + dof.upper() + "}$",
                     fontsize=16, fontweight='bold')
        custom_marker = Line2D([0], [0], marker='o', color='red',
                                 markerfacecolor='none', markeredgecolor='red',
                                 markeredgewidth=3, markersize=12,
                                 linestyle='None', label='Feature')
        ax.legend(handles=[custom_marker], loc='upper right')

    # --- RGB and Optical Flow Subplot at cell (0,3) ---
    # Create an inner subgridspec with 3 rows: 
    # Row 0: RGB image; Row 1: Optical Flow image; Row 2: 6-DoF text (placed outside optical flow)
    frame_ID = f"{frame_idx:06d}"
    # Update paths as needed.
    rgb_path = f"/scratch/chuan166/kitti/dataset/sequences/{opt.data_prefix}/image_2/{frame_ID}.png"
    flow_path = f"/scratch/chuan166/kitti/dataset/opticalflow/flow_{opt.data_prefix}/flow_{frame_ID}.png"
    try:
        rgb_img = Image.open(rgb_path).convert("RGB")
        flow_img = Image.open(flow_path).convert("RGB")
        # Optionally add a border to avoid cropping issues.
        rgb_img = ImageOps.expand(rgb_img, border=10, fill='black')
        flow_img = ImageOps.expand(flow_img, border=10, fill='black')
    except Exception as e:
        print(f"Error loading images for frame {frame_ID}: {e}")
        continue

    sub_gs = gs[0, 3].subgridspec(3, 1, height_ratios=[1, 1, 0.3])
    ax_rgb = fig.add_subplot(sub_gs[0])
    ax_flow = fig.add_subplot(sub_gs[1])
    ax_text = fig.add_subplot(sub_gs[2])
    ax_rgb.imshow(rgb_img)
    ax_rgb.axis('off')
    ax_rgb.set_title("RGB", fontsize=16, fontweight='bold')
    ax_flow.imshow(flow_img)
    ax_flow.axis('off')
    ax_flow.set_title("Optical Flow", fontsize=16, fontweight='bold')
    ax_text.axis('off')
    # Prepare six-DoF text annotation (all six components) to be displayed outside optical flow.
    dof_text = (f"Label: ({labels_arr[frame_idx,0]:.3f}, {labels_arr[frame_idx,1]:.3f}, {labels_arr[frame_idx,2]:.3f}, "
                f"{labels_arr[frame_idx,3]:.3f}, {labels_arr[frame_idx,4]:.3f}, {labels_arr[frame_idx,5]:.3f})\n"
                f"Pred: ({predicts[frame_idx,0]:.3f}, {predicts[frame_idx,1]:.3f}, {predicts[frame_idx,2]:.3f}, "
                f"{predicts[frame_idx,3]:.3f}, {predicts[frame_idx,4]:.3f}, {predicts[frame_idx,5]:.3f})")
    ax_text.text(0.5, 0.5, dof_text, transform=ax_text.transAxes,
                 ha='center', va='center', color='black', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))
    ax_text.set_title("6-DoF Info", fontsize=16, fontweight='bold')

    # --- Latent Space Subplots (3D) for row 1, columns 0-2 (roll, pitch, yaw) ---
    for dof in ['roll', 'pitch', 'yaw']:
        pos = latent_positions[dof]
        ax = fig.add_subplot(gs[pos[0], pos[1]], projection='3d')
        pca_result = pca_dict[dof]
        lab = labels_dict[dof]
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                   c=lab, cmap='viridis', alpha=0.9)
        current_point = pca_result[frame_idx]
        ax.text(current_point[0], current_point[1], current_point[2],
                '○', color='red', fontsize=24, ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=3, foreground='red')])
        ax.set_title(f"KITTI {opt.data_prefix} " + r"$\mathbf{" + dof.upper() + "}$",
                     fontsize=16, fontweight='bold')
        custom_marker = Line2D([0], [0], marker='o', color='red',
                                 markerfacecolor='none', markeredgecolor='red',
                                 markeredgewidth=3, markersize=12,
                                 linestyle='None', label='Feature')
        ax.legend(handles=[custom_marker], loc='upper right')

    # --- Trajectory Plot Subplot (2D) at cell (1,3) ---
    ax_traj = fig.add_subplot(gs[1, 3])
    # Plot full ground truth trajectory in green (static).
    ax_traj.plot(xl, zl, linewidth=3, color='g', label='Ground Truth (2D)')
    # Plot predicted trajectory only up to the current frame (red line).
    ax_traj.plot(xp[:frame_idx+1], zp[:frame_idx+1], linewidth=1, color='r', label='Predictions (2D)')
    # Mark the current prediction point with an 'rx' marker.
    ax_traj.plot(xp[frame_idx], zp[frame_idx], 'rx', markersize=8, label='Pred Current')
    ax_traj.set_xlabel('X')
    ax_traj.set_ylabel('Z')
    ax_traj.set_title(f"KITTI {opt.data_prefix} Trajectory (2D)", fontweight='bold')
    ax_traj.legend()
    ax_traj.set_aspect('equal', adjustable='datalim')

    # Adjust layout to avoid right-side cropping.
    plt.tight_layout()
    plt.subplots_adjust(right=0.98)
    save_path = os.path.join(data_dir, f'vocal_latent_3D_frame_{frame_idx:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved combined frame {frame_idx} to {save_path}")

print("All combined latent space frames saved.")

# ----- Video Creation -----
# After all frames are saved, compile them into a video using OpenCV.
frame_pattern = os.path.join(data_dir, "vocal_latent_3D_frame_*.png")
frame_files = sorted(glob.glob(frame_pattern))
if not frame_files:
    print("No frames found for video creation!")
else:
    # Read the first frame to get dimensions.
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    # height, width = 2364, 1195
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(data_dir, f"demo_{opt.data_prefix}.mp4")
    fps = 10  # adjust frame rate as needed
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        if not os.path.exists(frame_file):
            print(f"File not found: {frame_file}. Ending loop.")
            continue
        img = cv2.imread(frame_file)
        img = cv2.resize(img, (width, height))
        if img is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        video.write(img)
    video.release()
    print("Video saved as", video_path)

