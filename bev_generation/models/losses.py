# /bev_predictor/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVLoss(nn.Module):

    def __init__(self, 
                 alpha_weight=1.0,
                 rgb_weight=0.5,
                 smoothness_weight=0.1):
        super().__init__()
        

        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.smoothness_weight = smoothness_weight

    def compute_bev_loss(self, pred_bev, target_bev):


        pred_rgb, pred_alpha = pred_bev[:, :3], pred_bev[:, 3:]
        target_rgb, target_alpha = target_bev[:, :3], target_bev[:, 3:]


        alpha_loss = F.binary_cross_entropy(pred_alpha, target_alpha)


        road_mask = target_alpha > 0.5
        rgb_loss = F.mse_loss(pred_rgb * road_mask, target_rgb * road_mask)


        diff_x = torch.abs(pred_alpha[:, :, :, :-1] - pred_alpha[:, :, :, 1:])
        diff_y = torch.abs(pred_alpha[:, :, :-1, :] - pred_alpha[:, :, 1:, :])
        smoothness_loss = (diff_x.mean() + diff_y.mean()) / 2.0


        total_loss = (self.alpha_weight * alpha_loss +
                     self.rgb_weight * rgb_loss +
                     self.smoothness_weight * smoothness_loss)

        return total_loss, {
            'alpha_loss': alpha_loss.item(),
            'rgb_loss': rgb_loss.item(),
            'smoothness_loss': smoothness_loss.item()
        }

    def forward(self, predictions, targets):


        device = predictions['bev'].device
        predictions = {k: v.to(device) for k, v in predictions.items()}
        targets = {k: v.to(device) for k, v in targets.items()}


        total_loss, loss_components = self.compute_bev_loss(
            predictions['bev'], 
            targets['bev_view']
        )
        
        return total_loss, loss_components
