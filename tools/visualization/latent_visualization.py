
def plot_2d_projection(features_2d: np.ndarray,
                       labels: np.ndarray,
                       save_path: str,
                       title: str = 'Feature Projection') -> None:
    """
    Scatter‐plot 2D features colored by the underlying true angle.
    Expects labels[:,0]=sin, labels[:,1]=cos.
    """
    sin_vals = labels[:, 0]
    cos_vals = labels[:, 1]
    angles = np.arctan2(sin_vals, cos_vals)  # note: sin first, then cos

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=angles, cmap='hsv', alpha=0.7, s=15
    )
    plt.colorbar(scatter, label='angle (rad)')
    plt.xlabel('Feature Dim 1')
    plt.ylabel('Feature Dim 2')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_3d_sphere(coeffs_3d: np.ndarray, 
                   labels: np.ndarray,
                   save_path: str,
                   title: str = '3D Sphere Projection') -> None:
    """
    Plot 3D coefficients on a sphere, colored by azimuth and elevation.
    Similar to 2D projection but in 3D space.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = coeffs_3d[:, 0], coeffs_3d[:, 1], coeffs_3d[:, 2]
    
    # Extract azimuth and elevation from labels
    sin_az, cos_az, el = labels[:, 0], labels[:, 1], labels[:, 2]
    
    # Calculate azimuth angle for hue
    azimuth = np.arctan2(sin_az, cos_az)  # Range: -π to π
    
    # Normalize azimuth to [0, 1] for HSV hue (0 to 2π mapped to 0-1)
    azimuth_norm = (azimuth + np.pi) / (2 * np.pi)  # Map [-π, π] to [0, 1]
    
    # Find elevation range for normalization
    el_min = np.min(el)
    el_max = np.max(el)
    print(f"Elevation range: {el_min:.3f} to {el_max:.3f} radians")
    print(f"Elevation range: {el_min * 180/np.pi:.1f}° to {el_max * 180/np.pi:.1f}°")
    
    # Normalize elevation to [0, 1] for brightness/saturation
    if el_max > el_min:
        el_norm = (el - el_min) / (el_max - el_min)
    else:
        el_norm = np.ones_like(el) * 0.5  # If all elevations are the same
    
    # Create colors using HSV colormap
    # Use azimuth for hue, elevation for brightness
    colors = plt.cm.hsv(azimuth_norm)
    
    # Modulate brightness based on elevation
    # Higher elevation -> brighter, lower elevation -> darker
    for i in range(len(colors)):
        # Convert to HSV, modify brightness, convert back
        brightness_factor = 0.3 + 0.7 * el_norm[i]  # Range from 0.3 to 1.0
        colors[i, :3] *= brightness_factor  # Darken RGB channels
    
    scatter = ax.scatter(x, y, z, c=colors, alpha=0.8, s=20)
    
    # Create custom colorbar for azimuth
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Azimuth (rad)', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Coeff X', fontsize=12)
    ax.set_ylabel('Coeff Y', fontsize=12) 
    ax.set_zlabel('Coeff Z', fontsize=12)
    ax.set_title(f'{title}\n(Color: Azimuth as hue, Elevation as brightness)', fontsize=14)
    
    # Add text box with elevation info
    textstr = f'Elevation range: {el_min*180/np.pi:.1f}° to {el_max*180/np.pi:.1f}°\n(Brightness: darker=lower, brighter=higher)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()