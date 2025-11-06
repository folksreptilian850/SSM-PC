# /bev_predictor/models/loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetDetectionLoss(nn.Module):
    def __init__(self, 
                 visibility_weight=1.0,
                 position_weight=2.0,
                 attention_weight=0.5,
                 pos_weight=5.0):
        super().__init__()
        self.visibility_weight = visibility_weight
        self.position_weight = position_weight
        self.attention_weight = attention_weight
        self.pos_weight = pos_weight
        
    def compute_attention_loss(self, attention_maps, target_visibility):

        B, N, H, W = attention_maps.shape
        visibility_expanded = target_visibility.unsqueeze(-1).unsqueeze(-1)
        visibility_expanded = visibility_expanded.expand(-1, -1, H, W)
        

        invisible_penalty = torch.sum(
            attention_maps * (1 - visibility_expanded), dim=(2, 3)
        ).mean()
        

        entropy = -torch.sum(
            attention_maps * torch.log(attention_maps + 1e-10), dim=(2, 3)
        ).mean()
        
        return invisible_penalty + 0.1 * entropy

    def compute_position_loss(self, pred_positions, target_positions, visibility):

        visible_mask = (visibility > 0.5).unsqueeze(-1).expand(-1, -1, 2)
        
        if visible_mask.sum() > 0:
            pred_pos = pred_positions[visible_mask].view(-1, 2)
            target_pos = target_positions[visible_mask].view(-1, 2)
            

            l1_loss = F.smooth_l1_loss(pred_pos, target_pos, reduction='mean')
            

            pred_directions = F.normalize(pred_pos, dim=1)
            target_directions = F.normalize(target_pos, dim=1)
            direction_loss = (1 - torch.sum(pred_directions * target_directions, dim=1)).mean()
            
            return l1_loss + 0.2 * direction_loss
        else:
            return torch.tensor(0.0).to(pred_positions.device)

    def forward(self, predictions, targets):

        visibility_loss = F.binary_cross_entropy(
            predictions['object_visibility'],
            targets['object_visibility']
        )
        

        position_loss = self.compute_position_loss(
            predictions['object_relative_positions'],
            targets['object_relative_positions'],
            targets['object_visibility']
        )
        

        attention_loss = self.compute_attention_loss(
            predictions['attention_maps'],
            targets['object_visibility']
        )
        

        total_loss = (
            self.visibility_weight * visibility_loss +
            self.position_weight * position_loss +
            self.attention_weight * attention_loss
        )
        
        return total_loss, {
            'visibility_loss': visibility_loss.item(),
            'position_loss': position_loss.item(),
            'attention_loss': attention_loss.item()
        }
