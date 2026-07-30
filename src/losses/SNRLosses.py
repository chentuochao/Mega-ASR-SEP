import torch
import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR


class SNRLosses(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        if name == 'sisdr':
            self.loss_fn = SingleSrcNegSDR('sisdr')
        elif name == 'snr':
            self.loss_fn = SingleSrcNegSDR('snr')
        elif name == 'sdsdr':
            self.loss_fn = SingleSrcNegSDR('sdsdr')
        else:
            assert 0, f"Invalid loss function used: Loss {name} not found"

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        assert (torch.isnan(est).max() == 0), "Output tensor has nan!"
        assert (torch.isnan(gt).max() == 0), "GT tensor has nan!"

        est = est.reshape(B*C, T)
        gt = gt.reshape(B*C, T)
        
        return self.loss_fn(est_target=est, target=gt)


class SNRLPLoss(nn.Module):
    def __init__(self, name = "snr", neg_weight = 1) -> None:
        super().__init__()
        self.snr_loss = SNRLosses(name)
        #self.lp_loss = LogPowerLoss()
        self.lp_loss = nn.L1Loss(reduction = "none")#LogPowerLoss()
        # self.lp_loss2 = nn.L1Loss()#LogPowerLoss()
        self.neg_weight = neg_weight
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        input: (B, C, T) (B, C, T)
        """

        B, C, T = est.shape
        est = est.reshape(B*C, 1, T)
        gt = gt.reshape(B*C, 1, T)

        neg_loss = 0
        pos_loss = 0

        comp_loss = torch.zeros((est.shape[0]), device=est.device)
        
        mask = (torch.max(torch.max(torch.abs(gt), dim=2)[0], dim=1)[0] == 0)
        # print("mask", mask.shape, comp_loss.shape, est.shape, gt.shape)
        # If there's at least one negative sample
        if any(mask):
            est_neg, gt_neg = est[mask], gt[mask]
            
            neg_loss = self.lp_loss(est_neg, gt_neg).mean(dim=(1,2))
            # print("neg = ", self.lp_loss2(est_neg, gt_neg), neg_loss)
            # print("neg = ",neg_loss.shape, est_neg.shape, gt_neg.shape)
            comp_loss[mask] = neg_loss * self.neg_weight
            
        # If there's at least one positive sample
        if any((~ mask)):
            est_pos, gt_pos = est[~mask], gt[~mask]
            
            pos_loss = self.snr_loss(est_pos, gt_pos)
            # print("pos = ", pos_loss)
            # print("pos = ", pos_loss.shape, est_pos.shape, gt_pos.shape)
            # Compute_joint_loss
            comp_loss[~mask] = pos_loss
        
        return comp_loss