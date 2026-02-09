import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalDisentangler(nn.Module):
    """
    Disentangles features into Causal Content (C) and Spurious Style (S).
    Includes a Combiner to reconstruct the original features or generate counterfactuals.
    """
    def __init__(self, feature_dim, content_dim=None):
        super(CausalDisentangler, self).__init__()
        self.feature_dim = feature_dim
        self.content_dim = content_dim if content_dim is not None else feature_dim // 2
        self.style_dim = feature_dim - self.content_dim
        
        self.content_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim)
        )
        
        self.style_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, self.style_dim),
            nn.BatchNorm1d(self.style_dim)
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        c = self.content_encoder(x)
        s = self.style_encoder(x)
        return c, s
    
    def reconstruct(self, c, s):
        combined = torch.cat([c, s], dim=1)
        return self.combiner(combined)

class CausalDebiasedMultiViewClustering(nn.Module):
    """
    Revised Causal Module using Disentanglement and Counterfactual Generation.
    Strategy:
    1. Disentangle z -> (c, s)
    2. Swap 's' to generate Counterfactual z_cf
    3. Ensure c is invariant to this swap.
    """
    def __init__(self, num_views, feature_dim, device):
        super(CausalDebiasedMultiViewClustering, self).__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.device = device
     
        self.disentanglers = nn.ModuleList([
            CausalDisentangler(feature_dim) for _ in range(num_views)
        ])
        
    def forward(self, view_features_list, return_counterfactuals=True):
        """
        Args:
            view_features_list: List of [B, D]
        Returns:
            c_list: Content features [B, C_dim]
            c_cf_list: Counterfactual Content features
            z_rec_list: Reconstructed original features
            z_cf_list: Counterfactual features (useful for other tasks, or debugging)
        """
        c_list = []
        s_list = []
        z_rec_list = []
        
        for v in range(self.num_views):
            z = view_features_list[v]
            dis = self.disentanglers[v]
            
            c, s = dis(z)
            z_rec = dis.reconstruct(c, s)
            
            c_list.append(c)
            s_list.append(s)
            z_rec_list.append(z_rec)
            
        if not return_counterfactuals:
            return c_list, None, z_rec_list, None

        c_cf_list = []
        z_cf_list = []
        
        for v in range(self.num_views):
            dis = self.disentanglers[v]
            c = c_list[v]
            s = s_list[v]

            idx = torch.randperm(s.size(0))
            lambda_mix = 0.4
            s_mixed = (1 - lambda_mix) * s + lambda_mix * s[idx]

            z_cf = dis.reconstruct(c, s_mixed)
            z_cf_list.append(z_cf)
            
            c_extracted, _ = dis(z_cf)
            c_cf_list.append(c_extracted)
            
        return c_list, c_cf_list, z_rec_list, z_cf_list

class CausalContrastiveLoss(nn.Module):
    """
    Loss function for Causal Disentanglement.
    Includes:
    1. Content Alignment Loss (on Content C)
    2. Invariance Loss (C vs C_cf)
    3. Reconstruction Loss (Z vs Z_rec)
    """
    def __init__(self, temperature=0.5):
        super(CausalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.mse = nn.MSELoss()
        
    def forward(self, view_features, c_list, c_cf_list, z_rec_list):
        if c_list is None:
            return torch.tensor(0.0).to(view_features[0].device)
            
        device = view_features[0].device
        num_views = len(view_features)
        
        loss_rec = 0.0
        for v in range(num_views):
            loss_rec += self.mse(view_features[v], z_rec_list[v])
        loss_rec /= num_views

        loss_inv = 0.0
        if c_cf_list is not None:
            for v in range(num_views):
                loss_inv += self.mse(c_list[v].detach(), c_cf_list[v])
            loss_inv /= num_views

        loss_align = 0.0
        for i in range(num_views):
            for j in range(i + 1, num_views):

                z_i = F.normalize(c_list[i], dim=1)
                z_j = F.normalize(c_list[j], dim=1)
                
                sim = torch.matmul(z_i, z_j.T) / self.temperature

                labels = torch.arange(z_i.size(0)).to(device)
                l1 = F.cross_entropy(sim, labels)
                l2 = F.cross_entropy(sim.T, labels)
                loss_align += (l1 + l2) / 2
                
        loss_align /= (num_views * (num_views - 1) / 2)

        return loss_rec, loss_inv, loss_align

class ViewInvarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tensor(0.0)

