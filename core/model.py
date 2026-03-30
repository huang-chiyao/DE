import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
import timm  # For loading pre-trained Vision Transformer model
from scipy.optimize import linear_sum_assignment
import time

def check_orthogonality(evecs, tol=1e-3):
    
    # Normalize each eigenvector (in case they're not already unit norm)
    evecs_norm = evecs / evecs.norm(dim=0, keepdim=True).clamp_min(1e-8)
    
    # Compute pairwise cosine similarity (Gram matrix)
    gram = torch.matmul(evecs_norm.t(), evecs_norm)
    
    # Compare to identity
    identity = torch.eye(gram.size(0), device=gram.device)
    diff = torch.abs(gram - identity)
    
    max_offdiag = diff[~torch.eye(diff.size(0), dtype=bool, device=diff.device)].max()
    mean_offdiag = diff[~torch.eye(diff.size(0), dtype=bool, device=diff.device)].mean()
    
    print(f"Max off-diagonal deviation: {max_offdiag.item():.4e}")
    print(f"Mean off-diagonal deviation: {mean_offdiag.item():.4e}")
    
    # Print the pairwise cosine similarities (off-diagonal)
    print("\nPairwise cosine similarities (off-diagonal):")
    n = gram.size(1)
    if n > 10:
        print(f"Note: Only showing first 10 of {n} eigenvectors for brevity.")
        n = 10
    for i in range(n):
        for j in range(i + 1, n):
            print(f"cos(v{i}, v{j}) = {gram[i, j].item():+.4e}")
    
    if max_offdiag < tol:
        print("\n✅ Eigenvectors are approximately orthogonal.")
    else:
        print("\n⚠️ Eigenvectors are not perfectly orthogonal.")
    
    return gram


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=6, dropout=None, width_per_group=64):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet50': [resnet50, 2048]
}


class Encoder(nn.Module):
    def __init__(self, name='resnet18'):
        super(Encoder, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()

    def forward(self, x):
        feat = self.encoder(x)
        return feat


def create_base_model(model_type, in_channels=2):
    """Create base model architecture"""
    if 'resnet18' in model_type:
        model = models.resnet18(pretrained=False)
        feature_dim = 512
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity()
        return model, feature_dim

    elif 'resnet50' in model_type:
        model = models.resnet50(pretrained=False)
        feature_dim = 2048
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity()
        return model, feature_dim

    elif 'ViT' in model_type:
        model = timm.create_model('vit_base_patch16_224',
                                  pretrained=False,
                                  in_chans=in_channels)
        model.reset_classifier(num_classes=0)
        feature_dim = 768
        return model, feature_dim
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class Regressors(nn.Module):
    def __init__(self, feature_dim=512, output_dim=6, decoder_loss_list=None, num_cls=None, opt=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.opt = opt
        if decoder_loss_list is None:
            decoder_loss_list = ["l1"] * output_dim
        assert len(decoder_loss_list) == output_dim, \
            f"decoder_loss_list length {len(decoder_loss_list)} must match output_dim {output_dim}"

        self.decoders = nn.ModuleList()
        for i, loss in enumerate(decoder_loss_list):
            if loss.lower() == "ce":
                print(f"Decoder {i}: CE classifier with {num_cls} classes")
                self.decoders.append(nn.Linear(self.feature_dim, num_cls))
            elif loss.lower() == "l1":
                print(f"Decoder {i}: regression (L1) head")
                self.decoders.append(nn.Linear(self.feature_dim, 1))
            else:
                raise ValueError(f"Unsupported decoder loss type: {loss}")

    def load_eigen_params(self, V, mu, scale):
        """
        V : Tensor (feature_dim, output_dim)
        mu : Tensor (feature_dim,)
        scale : Tensor (output_dim,)
        """
        with torch.no_grad():
            for i, dec in enumerate(self.decoders):
                # if self.opt.output_list[i].lower() in ['classification', 'class', 'cls', 'id']:
                    # For classification heads, we do not set eigen params
                    # continue
                assert isinstance(dec, nn.Linear)
                v_i = V[:, i].unsqueeze(0)   # (1, D)
                s_i = scale[i]
                # set weights
                dec.weight.copy_(s_i * v_i)   # (1, D)
                # set bias
                dec.bias.copy_(torch.tensor([-s_i * v_i @ mu.T]))

    
    def forward(self, f_proj):
        preds = []
        for i, dec in enumerate(self.decoders):
            preds.append(dec(f_proj))
        return preds


class RNC_Encoder(nn.Module):
    def __init__(self, opt, in_channels=2):
        super(RNC_Encoder, self).__init__()

        self.model_type = opt.model

        # Create base model
        self.model, feature_dim = create_base_model(self.model_type, in_channels)
        self.feature_dim = feature_dim
        print(f"current feature dim: {feature_dim}")

        self.register_buffer("running_mean", torch.zeros(1, self.feature_dim))

    def forward(self, img, train=False):
        f = self.model(img)
        B = f.size(0)

        if B > 1 or train:
            batch_mean = f.mean(dim=0, keepdim=True)
            feat_mean = self.running_mean
        else:
            feat_mean = self.running_mean

        f_centered = f - feat_mean
        return f_centered, f, None


class Domain_Encoder(nn.Module):
    def __init__(self, opt, in_channels=2):
        super().__init__()

        print('currently use fix eigen')

        self.model_type = opt.model
        self.output_list = opt.output_list
        self.output_dimension_list = opt.output_dimension_list
        
        # Validate input consistency
        if len(self.output_list) != len(self.output_dimension_list):
            raise ValueError("output_list and output_dimension_list must have the same length")


        self.model, feature_dim = create_base_model(opt.model, in_channels)

        self.feature_dim = feature_dim

        # Calculate total number of eigenvectors needed
        self.total_eigen_dim = sum(self.output_dimension_list)
        print(f"model/total_eigen_dim: {self.total_eigen_dim}")
        print(f"model/output_list: {self.output_list}")
        print(f"model/output_dimension_list: {self.output_dimension_list}")

        # Hungarian alignment control
        self.h_align = not getattr(opt, 'no_hungarian_alignment', False)
        print("model/hungarian_alignment ", self.h_align)

        # Register buffers - updated to use total_eigen_dim
        self.register_buffer("prev_evecs", torch.zeros(self.feature_dim, self.total_eigen_dim))
        self.register_buffer("running_mean", torch.zeros(1, self.feature_dim))

        # Normalization
        self.normalize = opt.normalize_classification
        print("model/normalize_classification ", self.normalize)


    def forward(self, img, project=True, return_centered=False):
        # Extract raw features
        f = self.model(img)

        # Apply projection head if it exists
        if hasattr(self, 'projection_head') and self.projection_head is not None:
            f = self.projection_head(f)

        f_centered = f - self.running_mean
        
        # Initialize lists for projections and coefficients
        f_proj_list = []
        proj_coeffs_list = []
        
        # Track current eigenvector index
        evec_start_idx = 0
        
        output_list, output_dimension_list = self.output_list, self.output_dimension_list
            
        # Process each output type based on output_dimension_list
        for i, (output_type, num_dims) in enumerate(zip(output_list, output_dimension_list)):
            # Get eigenvectors for current output type
            evec_end_idx = evec_start_idx + num_dims
            current_evecs = self.prev_evecs[:, evec_start_idx:evec_end_idx]  # (feature_dim, num_dims)
            
            # Project features onto current eigenvectors
            coeffs = torch.matmul(f_centered, current_evecs)  # (batch_size, num_dims)
            
            if num_dims == 1:
                # Single eigenvector: coefficient is the projection length
                proj_coeff = coeffs.squeeze(-1)  # (batch_size,)
                # Reconstruct projection in feature space
                f_proj_current = coeffs * current_evecs.t()  # (batch_size, feature_dim)
            else:
                # Multiple eigenvectors: coefficient is L2 norm of projection
                proj_coeff = torch.norm(coeffs, dim=1)  # (batch_size,)
                # Reconstruct projection in feature space (project into subspace)
                f_proj_current = torch.matmul(coeffs, current_evecs.t())  # (batch_size, feature_dim)
            
            # Apply normalization if specified and this is a classification output
            if self.normalize and output_type.lower() in ['id', 'cls']:
                f_proj_current = f_proj_current / torch.norm(f_proj_current, dim=1, keepdim=True)
            
            f_proj_list.append(f_proj_current)
            proj_coeffs_list.append(proj_coeff)
            
            # Move to next set of eigenvectors
            evec_start_idx = evec_end_idx
        
        # Stack results
        # f_proj: (batch_size, len(output_list), feature_dim)
        f_proj = torch.stack(f_proj_list, dim=1)
        
        # proj_coeffs: (batch_size, len(output_list))
        proj_coeffs = torch.stack(proj_coeffs_list, dim=1)

        if not return_centered:
            return f_proj, f, proj_coeffs
        else:
            return f_proj, f_centered, proj_coeffs

    def get_eigenvectors_and_projection(self, f, train=True, epoch=None, update=True):
        B = f.size(0)
        curr_mean = f.mean(dim=0, keepdim=True)
        prev_running_mean = self.running_mean.clone()
        
        f_centered = f - self.running_mean
        cosine_sims = None
            
        if update:
            # Compute SVD
            
            U, S, Vh = torch.linalg.svd(f_centered.cuda(), full_matrices=False)
            U = U.to("cpu")
            S = S.to("cpu")
            Vh = Vh.to("cpu")
            current_evecs = Vh[:self.total_eigen_dim, :].t()
            current_eigenvals = S[:self.total_eigen_dim] 
            
            # Log mean changes
            mean_change = torch.norm(self.running_mean - prev_running_mean).item()
            log_dict = {
                "encoder_eigenvector/epoch": epoch,
                "encoder_eigenvector/mean_change": mean_change
            }
            
            # Eigenvector alignment
            if torch.any(self.prev_evecs != 0) and self.h_align:
                similarity_matrix = torch.abs(torch.matmul(self.prev_evecs.t(), current_evecs))
                cost_matrix = 1 - similarity_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                aligned_evecs = torch.empty_like(current_evecs)
                aligned_eigenvals = torch.empty_like(current_eigenvals)
                
                for i, j in zip(row_ind, col_ind):
                    candidate = current_evecs[:, j]
                    if torch.dot(self.prev_evecs[:, i], candidate) < 0:
                        candidate = -candidate
                    aligned_evecs[:, i] = candidate
                    aligned_eigenvals[i] = current_eigenvals[j]
                
                current_evecs = aligned_evecs
                current_eigenvals = aligned_eigenvals

            assert current_evecs.shape == (self.feature_dim, self.total_eigen_dim), \
                f"current_evecs shape {current_evecs.shape} does not match expected {(self.feature_dim, self.total_eigen_dim)}"
            
            # Orthogonality check
            check_orthogonality(current_evecs)
            
            # Cosine similarity logging
            if torch.any(self.prev_evecs != 0):
                cosine_sims = []
                evec_idx = 0
                
                # Calculate per-output type mean cosine similarity
                for output_idx, (output_type, num_dims) in enumerate(zip(self.output_list, self.output_dimension_list)):
                    output_cosine_sims = []
                    for dim_idx in range(num_dims):
                        cos_sim = torch.dot(self.prev_evecs[:, evec_idx], current_evecs[:, evec_idx])
                        cos_sim_abs = abs(cos_sim.item())
                        cosine_sims.append(cos_sim_abs)
                        output_cosine_sims.append(cos_sim_abs)
                        evec_idx += 1
                    
                    # Per-output type mean only
                    if output_cosine_sims:
                        log_dict[f"encoder_eigenvector/{output_type}_cosine_similarity_mean"] = sum(output_cosine_sims) / len(output_cosine_sims)
                
                # Overall statistics (keep existing)
                log_dict["encoder_eigenvector/cosine_similarity_mean"] = sum(cosine_sims) / len(cosine_sims)
            
            # Update stored eigenvectors
            self.prev_evecs = current_evecs.clone().detach()
            
            # Explained variance (global only)
            total_energy = S.sum()
            explained_variance_ratios = current_eigenvals / total_energy
            for i, ratio in enumerate(explained_variance_ratios):
                log_dict[f"encoder_eigenvector/explained_variance_ratio_dim_{i + 1}"] = ratio.item()
            log_dict["encoder_eigenvector/explained_variance_ratio_total"] = explained_variance_ratios.sum().item()
            
            wandb.log(log_dict)
        
        # Calculate projection coefficients and their std
        proj_coeffs = torch.matmul(f_centered, self.prev_evecs)
        proj_std = torch.std(proj_coeffs, dim=0).cpu().numpy()
        
        return proj_std, cosine_sims
