import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def parse_option():
    parser = argparse.ArgumentParser('plotting')
    parser.add_argument('--data_prefix', type=str, default='00', help='dataset prefix')
    opt = parser.parse_args()
    return opt

opt = parse_option()

# Load results from the NPZ file
results = np.load(f'results/results_{opt.data_prefix}.npz')

# Extract DOF-specific predictions from the loaded results
predicts = np.zeros((len(results['label_x']), 6))  # 6-DoF: x, y, z, roll, pitch, yaw
predicts[:, 0] = results['output_x'].flatten()
predicts[:, 1] = results['output_y'].flatten()
predicts[:, 2] = results['output_z'].flatten()
predicts[:, 3] = results['output_roll'].flatten()
predicts[:, 4] = results['output_pitch'].flatten()
predicts[:, 5] = results['output_yaw'].flatten()

# Extract the corresponding ground truth labels
labels = np.zeros((len(results['label_x']), 6))
labels[:, 0] = results['label_x'].flatten()
labels[:, 1] = results['label_y'].flatten()
labels[:, 2] = results['label_z'].flatten()
labels[:, 3] = results['label_roll'].flatten()
labels[:, 4] = results['label_pitch'].flatten()
labels[:, 5] = results['label_yaw'].flatten()

# Initialize transformation matrices
labels_mat = np.zeros((len(labels), 4, 4))
predicts_mat = np.zeros((len(predicts), 4, 4))
new_labels = np.zeros((len(labels), 4, 4))
new_predicts = np.zeros((len(predicts), 4, 4))

# Accumulate transformations for labels
current_p = np.eye(4)
for i in range(len(labels)):
    rl = R.from_rotvec(labels[i, 3:6])  # Rotation vector to matrix
    labels_mat[i, 0:3, 0:3] = rl.as_matrix()
    labels_mat[i, 0:3, 3] = labels[i, 0:3]  # Translation part
    labels_mat[i, 3, 3] = 1.0

    current_p = np.matmul(current_p, labels_mat[i])
    new_labels[i] = current_p

# Accumulate transformations for predictions
current_p = np.eye(4)
for i in range(len(predicts)):
    rp = R.from_rotvec(predicts[i, 3:6])  # Rotation vector to matrix
    predicts_mat[i, 0:3, 0:3] = rp.as_matrix()
    predicts_mat[i, 0:3, 3] = predicts[i, 0:3]  # Translation part
    predicts_mat[i, 3, 3] = 1.0

    current_p = np.matmul(current_p, predicts_mat[i])
    new_predicts[i] = current_p

# Extract x, y, z coordinates from transformation matrices
xl, yl, zl = new_labels[:, 0, 3], new_labels[:, 2, 3], new_labels[:, 1, 3]
xp, yp, zp = new_predicts[:, 0, 3], new_predicts[:, 2, 3], new_predicts[:, 1, 3]

# Plot the 3D trajectory
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Plot ground truth trajectory in 3D
ax_3d.plot(xl, yl, zl, linewidth=3, color='g', label='Ground Truth')
# Plot predicted trajectory in 3D
ax_3d.plot(xp, yp, zp, linewidth=1, color='r', label='Predictions')

# Set axis labels, legend, and bold title for 3D plot
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Z')
ax_3d.set_zlabel('Y')
ax_3d.legend()
ax_3d.set_title(f"KITTI {opt.data_prefix}", fontweight='bold')

# Set the same scale for X, Y, and Z axes
ax_3d.set_aspect('equal', adjustable='datalim')

# Save 3D plot
plt.savefig(f'trajectory/vocal_plot_3d_{opt.data_prefix}.png')

# Plot the 2D x-z trajectory
fig_2d = plt.figure()
ax_2d = fig_2d.add_subplot(111)

# Plot ground truth and predicted trajectories in 2D (x-z)
ax_2d.plot(xl, yl, linewidth=3, color='g', label='Ground Truth (2D)')
ax_2d.plot(xp, yp, linewidth=1, color='r', label='Predictions (2D)')

# Set axis labels, legend, and bold title for 2D plot
ax_2d.set_xlabel('X')
ax_2d.set_ylabel('Z')
ax_2d.legend()
ax_2d.set_title(f"KITTI {opt.data_prefix}", fontweight='bold')

# Set the same scale for X and Z axes
ax_2d.set_aspect('equal', adjustable='datalim')

# Save 2D plot
plt.savefig(f'trajectory/vocal_plot_2d_{opt.data_prefix}.png')
