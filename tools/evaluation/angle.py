import torch


def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch.
    This is a PyTorch implementation computed by myself for calculating
    R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    
    For the original numpy implementation in StarMap, you can refer to:
    https://github.com/xingyizhou/StarMap/blob/26223a6c766eab3c22cddae87c375150f84f804d/tools/EvalCls.py#L20
    """
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)


def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()
    
    # get elevation and inplane-rotation in the right format
    # R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    preds[:, 1] = preds[:, 1] - 180.
    preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1] - 180.
    targets[:, 2] = targets[:, 2] - 180.
    
    # change degrees to radians
    preds = preds * torch.pi / 180.
    targets = targets * torch.pi / 180.
    
    # get rotation matrix from euler angles
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    
    # compute the angle distance between rotation matrix in degrees
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / torch.pi
    return R_err


def calculate_circular_mae_split(pred_sin, pred_cos, true_sin, true_cos):
    """Calculate circular mean absolute error between predicted and true angles."""
    # Convert sin, cos to angles
    pred_angles = torch.atan2(pred_sin, pred_cos)
    true_angles = torch.atan2(true_sin, true_cos)
    
    # Calculate angular difference
    diff = pred_angles - true_angles
    
    # Wrap to [-pi, pi]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    # Return mean absolute error in radians
    return torch.abs(diff).mean(dim=0)


def calculate_circular_mae(pred_angle, true_angle):
    """Calculate circular mean absolute error between predicted and true angles."""
    # Calculate angular difference
    diff = pred_angle - true_angle
    
    # Wrap to [-pi, pi]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    # Return mean absolute error in radians
    return torch.abs(diff).mean(dim=0)


def calculate_mae(output_list, label_list):
    # Convert lists to torch tensors if they are not already
    if not isinstance(output_list, torch.Tensor):
        output_tensor = torch.tensor(output_list)
    else:
        output_tensor = output_list
    
    if not isinstance(label_list, torch.Tensor):
        label_tensor = torch.tensor(label_list)
    else:
        label_tensor = label_list

    # Calculate the Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(output_tensor - label_tensor), dim=0)
    return mae


def viewpoint_error_3dof(pred_angles, true_angles):
    """Calculate viewpoint error for 3DOF case (azimuth + elevation + roll).
    
    Args:
        pred_angles: Tensor of shape [batch_size, 3] containing [azimuth, elevation, roll] in radians
        true_angles: Tensor of shape [batch_size, 3] containing [azimuth, elevation, roll] in radians
    
    Returns:
        errors: Tensor of rotation errors in radians
    """
    # Use existing rotation_err function (expects degrees)
    errors = rotation_err(pred_angles * 180.0 / torch.pi, true_angles * 180.0 / torch.pi)
    return errors * torch.pi / 180.0  # Convert back to radians


def viewpoint_error_6dof(pred, true):
    """Calculate viewpoint error for 6DOF case (azimuth + elevation + roll), each split into "sin" and "cos".
    
    Args:
        pred: Tensor of shape [batch_size, 6] containing [sin_az, cos_az, sin_el, cos_el, sin_ro, cos_ro]
        true: Tensor of shape [batch_size, 6] containing [sin_az, cos_az, sin_el, cos_el, sin_ro, cos_ro]
    
    Returns:
        errors: Tensor of rotation errors in radians
    """
    # Extract components from pred tensor
    pred_sin_az = pred[:, 0]
    pred_cos_az = pred[:, 1]
    pred_sin_el = pred[:, 2]
    pred_cos_el = pred[:, 3]
    pred_sin_ro = pred[:, 4]
    pred_cos_ro = pred[:, 5]
    # Note: pred[:, 5] is unused in this implementation
    
    # Extract components from true tensor
    true_sin_az = true[:, 0]
    true_cos_az = true[:, 1]
    true_sin_el = true[:, 2]
    true_cos_el = true[:, 3]
    true_sin_ro = true[:, 4]
    true_cos_ro = true[:, 5]
    # Note: true[:, 5] is unused in this implementation
    
    # Convert predictions to euler angles
    pred_azi = torch.atan2(pred_sin_az, pred_cos_az)
    pred_ele = torch.atan2(pred_sin_el, pred_cos_el)
    pred_rol = torch.atan2(pred_sin_ro, pred_cos_ro)
    
    # Convert targets to euler angles
    true_azi = torch.atan2(true_sin_az, true_cos_az)
    true_ele = torch.atan2(true_sin_el, true_cos_el)
    true_rol = torch.atan2(true_sin_ro, true_cos_ro)
    
    # Stack angles: [azimuth, elevation, roll]
    pred_angles = torch.stack([pred_azi, pred_ele, pred_rol], dim=1)
    true_angles = torch.stack([true_azi, true_ele, true_rol], dim=1)
    
    # Use existing rotation_err function
    errors = rotation_err(pred_angles * 180.0 / torch.pi, true_angles * 180.0 / torch.pi)
    return errors * torch.pi / 180.0  # Convert back to radians

def viewpoint_error_2dof(pred_angles, true_angles):
    """Calculate viewpoint error for 2DOF case (azimuth + elevation only).
    
    Args:
        pred_angles: Tensor of shape [batch_size, 2] containing [azimuth, elevation] in radians
        true_angles: Tensor of shape [batch_size, 2] containing [azimuth, elevation] in radians
    
    Returns:
        errors: Tensor of rotation errors in radians
    """
    # Pad with zeros for roll dimension to make it 3DOF
    batch_size = pred_angles.shape[0]
    zeros = torch.zeros(batch_size, 1, device=pred_angles.device, dtype=pred_angles.dtype)
    
    # Stack to create [azimuth, elevation, roll=0]
    pred_angles_3d = torch.cat([pred_angles, zeros], dim=1)
    true_angles_3d = torch.cat([true_angles, zeros], dim=1)
    
    # Use existing rotation_err function (expects degrees)
    errors = rotation_err(pred_angles_3d * 180.0 / torch.pi, true_angles_3d * 180.0 / torch.pi)
    return errors * torch.pi / 180.0  # Convert back to radians


def viewpoint_error_4dof(pred, true):
    """Calculate viewpoint error for 4DOF case (azimuth + elevation), each split into "sin" and "cos".
    
    Args:
        pred: Tensor of shape [batch_size, 4] containing [sin_az, cos_az, sin_el, cos_el]
        true: Tensor of shape [batch_size, 4] containing [sin_az, cos_az, sin_el, cos_el]
    
    Returns:
        errors: Tensor of rotation errors in radians
    """
    # Extract components from pred tensor
    pred_sin_az = pred[:, 0]
    pred_cos_az = pred[:, 1]
    pred_sin_el = pred[:, 2]
    pred_cos_el = pred[:, 3]
    
    # Extract components from true tensor
    true_sin_az = true[:, 0]
    true_cos_az = true[:, 1]
    true_sin_el = true[:, 2]
    true_cos_el = true[:, 3]
    
    # Convert predictions to euler angles
    pred_azi = torch.atan2(pred_sin_az, pred_cos_az)
    pred_ele = torch.atan2(pred_sin_el, pred_cos_el)
    
    # Convert targets to euler angles
    true_azi = torch.atan2(true_sin_az, true_cos_az)
    true_ele = torch.atan2(true_sin_el, true_cos_el)
    
    # Create zeros for roll dimension
    batch_size = pred.shape[0]
    zeros = torch.zeros(batch_size, device=pred.device, dtype=pred.dtype)
    
    # Stack angles: [azimuth, elevation, roll=0]
    pred_angles = torch.stack([pred_azi, pred_ele, zeros], dim=1)
    true_angles = torch.stack([true_azi, true_ele, zeros], dim=1)
    
    # Use existing rotation_err function
    errors = rotation_err(pred_angles * 180.0 / torch.pi, true_angles * 180.0 / torch.pi)
    return errors * torch.pi / 180.0  # Convert back to radians