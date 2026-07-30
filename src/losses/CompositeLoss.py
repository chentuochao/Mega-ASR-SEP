import torch
import torch.nn as nn
import src.utils as utils

class CompositeLoss(nn.Module):
    def __init__(self, losses) -> None:
        super().__init__()
        self.losses = []
        self.weights = []

        for loss_desc in losses:
            if 'params' not in loss_desc:
                loss_desc['params'] = {}
            if 'weight' not in loss_desc:
                loss_desc['weight'] = 1
            
            loss = utils.import_attr(loss_desc['loss'])(**loss_desc['params'])
            self.losses.append(loss)
            self.weights.append(loss_desc['weight'])
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        loss = None
        for weight, loss_fn in zip(self.weights, self.losses):
            L = loss_fn(est, gt)
            
            # Take mean over element in a batch independently
            L = L.mean()
            
            if loss is None:
                loss = L * weight
            else: 
                loss += L * weight
        return loss

class PosNegLoss(nn.Module):
    def __init__(self, pos_loss, pos_loss_params, neg_loss, neg_loss_params, pos_weight=1, neg_weight=1) -> None:
        super().__init__()
        self.neg_loss = utils.import_attr(neg_loss)(**neg_loss_params)
        self.pos_loss = utils.import_attr(pos_loss)(**pos_loss_params)
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape
        est = est.reshape(B, C, T)
        gt = gt.reshape(B, C, T)
        comp_loss = torch.zeros((est.shape[0]), device=est.device)
        
        mask = (torch.max(torch.max(torch.abs(gt), dim=2)[0], dim=1)[0] == 0)
        
        # If there's at least one negative sample
        if any(mask):
            est_neg, gt_neg = est[mask], gt[mask]
            
            neg_loss = self.neg_loss(est_neg, gt_neg)
            comp_loss[mask] = neg_loss * self.neg_weight
            
        # If there's at least one positive sample
        if any((~ mask)):
            est_pos, gt_pos = est[~mask], gt[~mask]
            
            comp_loss[~mask] = self.pos_loss(est_pos, gt_pos) * self.pos_weight        

        return comp_loss