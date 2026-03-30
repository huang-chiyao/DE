import torch
import torch.nn as nn
import torch.nn.functional as F
        
class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l2':
            return (labels[:, None, :] - labels[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)

class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels, temperature=None):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        
        if temperature is not None:
            t = temperature
        else:
            t = self.t

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            denom = (neg_mask * exp_logits).sum(dim=-1) + 1e-8  # epsilon for safety
            pos_log_probs = pos_logits - torch.log(denom)
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.feature_similarity = 'l2'

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # shapes: anchor_feature [A, D], contrast_feature [B, D] -> dist [A, B]
        diff = anchor_feature[:, None, :] - contrast_feature[None, :, :]
        if self.feature_similarity == 'l1':
            dist = diff.abs().sum(dim=-1)
        elif self.feature_similarity == 'l2':
            # Euclidean distance (add eps for stable gradient at 0)
            dist = diff.norm(2, dim=-1)
            # dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps)
        else:  # 'sq_l2'
            dist = (diff * diff).sum(dim=-1)

        # convert distance to similarity logits for softmax
        anchor_dot_contrast = -(dist) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        log_prob = torch.clip(log_prob, min=-1e6, max=1e6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DomainLoss(nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()

    def forward(self, features, eigenvectors):
        """
        Computes the loss:
            L = mean_{b, j, k} [ ((f_b - v_j)ᵀ (f_b - v_k) - 1)² ]
        
        Args:
            features (torch.Tensor): Tensor of shape [B, D]
            eigenvectors (torch.Tensor): Tensor of shape [K, D]
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # features: [B, D], eigenvectors: [K, D]
        # For each feature vector, compute the difference with every eigenvector.
        # Resulting tensor diff: [B, K, D]
        diff = features.unsqueeze(1) - eigenvectors.unsqueeze(0)
        
        # Compute the dot product between the difference vectors for every pair of eigenvectors.
        # One way to do this is to use torch.matmul.
        # diff has shape [B, K, D], so diff.transpose(1,2) has shape [B, D, K]
        # The resulting dot_products tensor has shape [B, K, K], where
        # dot_products[b, j, k] = (f_b - v_j) dot (f_b - v_k)
        dot_products = torch.matmul(diff, diff.transpose(1, 2))
        
        # Compute the squared error from the target value 1.
        loss = torch.mean((dot_products - 1) ** 2)
        
        return loss
    
class CELoss(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(CELoss, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Linear(feature_dim, num_classes)  # Example linear layer
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        return self.ce_loss(self.model(features), labels)