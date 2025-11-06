# models/toroidal_grid_cell.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GridCellNetwork(nn.Module):






    def __init__(
        self,
        place_cells,
        hd_cells,
        input_size=3,
        hidden_size=128,
        bottleneck_size=256,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.place_cells = place_cells
        self.hd_cells = hd_cells
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size

               
        self.lstm = nn.LSTM(
            input_size=input_size,                            
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

                     
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size, bias=False),
            nn.Dropout(dropout_rate)
        )

                                              
        self.place_predictor = nn.Linear(bottleneck_size, place_cells.n_cells)
        self.hd_predictor = nn.Linear(bottleneck_size, hd_cells.n_cells)

                                        
                                                                                              
        self.init_fc = nn.Linear(place_cells.n_cells + hd_cells.n_cells, hidden_size * 2)
                               
        self.h0_init = nn.Linear(place_cells.n_cells + hd_cells.n_cells, hidden_size)
        self.c0_init = nn.Linear(place_cells.n_cells + hd_cells.n_cells, hidden_size)

    def forward(self, velocity, init_pos, init_hd):






        # print("init_pos range:", init_pos.min().item(), init_pos.max().item())
        # print("init_place_act mean:", init_place_act.mean().item(), "std:", init_place_act.std().item())


        batch_size, seq_len = velocity.shape[:2]

                                
        init_place_act = self.place_cells.compute_activation(init_pos)  # [B, n_place]
        init_hd_act = self.hd_cells.compute_activation(init_hd)          # [B, n_hd]
        init_repr = torch.cat([init_place_act, init_hd_act], dim=-1)       # [B, n_place+n_hd]

                                    
        h0 = self.h0_init(init_repr).unsqueeze(0).contiguous()
        c0 = self.c0_init(init_repr).unsqueeze(0).contiguous()

                   
        lstm_out, _ = self.lstm(velocity, (h0, c0))  # [B, seq, hidden_size]

                          
        B, T, H = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(B * T, H)
        bottleneck_flat = self.bottleneck(lstm_out_flat)  # [B*T, bottleneck_size]
        place_logits_flat = self.place_predictor(bottleneck_flat)
        hd_logits_flat = self.hd_predictor(bottleneck_flat)

                        
        place_logits = place_logits_flat.reshape(B, T, -1)
        hd_logits = hd_logits_flat.reshape(B, T, -1)

        return {
            'place_logits': place_logits,  # [B, seq, n_place]
            'hd_logits': hd_logits,        # [B, seq, n_hd]
            'bottleneck': bottleneck_flat.reshape(B, T, -1),
            'lstm_out': lstm_out,
        }



class InitialStateRegularizedLoss(nn.Module):





    def __init__(self, initial_frames_weight=5.0, decay_factor=0.8):





        super().__init__()
        self.initial_frames_weight = initial_frames_weight
        self.decay_factor = decay_factor

    def forward(self, outputs, targets):
                       
        place_logits = outputs['place_logits']  # [B, seq, n_place]
        place_targets = targets['place_targets']  # [B, seq, n_place]
        
        hd_logits = outputs['hd_logits']  # [B, seq, n_hd]
        hd_targets = targets['hd_targets']  # [B, seq, n_hd]
        
        batch_size, seq_len = place_logits.shape[:2]
        
               
        total_place_loss = 0.0
        total_hd_loss = 0.0
        total_weight = 0.0
        
                      
        for t in range(seq_len):
                                      
            if t == 0:                   
                weight = self.initial_frames_weight * 2.0           
            elif t < 5:             
                weight = self.initial_frames_weight * (self.decay_factor ** (t-1))
            else:
                weight = 1.0             
            
                                
            place_log_softmax_t = F.log_softmax(place_logits[:, t], dim=-1)
            place_loss_t = F.kl_div(
                place_log_softmax_t,
                place_targets[:, t],
                reduction='batchmean'
            )
            
            hd_log_softmax_t = F.log_softmax(hd_logits[:, t], dim=-1)
            hd_loss_t = F.kl_div(
                hd_log_softmax_t,
                hd_targets[:, t],
                reduction='batchmean'
            )
            
                  
            total_place_loss += weight * place_loss_t
            total_hd_loss += weight * hd_loss_t
            total_weight += weight
        
                      
        avg_place_loss = total_place_loss / total_weight
        avg_hd_loss = total_hd_loss / total_weight
        total_loss = avg_place_loss + avg_hd_loss
        
                                            
        first_place_pred = F.log_softmax(place_logits[:, 0], dim=-1)
        first_place_target = place_targets[:, 0]
        init_consistency_loss = F.kl_div(
            first_place_pred,
            first_place_target,
            reduction='batchmean'
        ) * 2.0              
        
                               
        continuity_loss = 0.0
        if seq_len > 1:
            for t in range(1, min(6, seq_len)):                   
                continuity_weight = 1.0 - 0.15 * (t-1)                
                place_pred_prev = F.softmax(place_logits[:, t-1], dim=-1)
                place_pred_curr = F.softmax(place_logits[:, t], dim=-1)
                
                                      
                continuity_loss += continuity_weight * F.kl_div(
                    torch.log(place_pred_curr + 1e-10),
                    place_pred_prev,
                    reduction='batchmean'
                )
        
                 
        final_loss = total_loss + init_consistency_loss + 0.1 * continuity_loss
        
        return {
            'total': final_loss,
            'place': avg_place_loss.detach().cpu().item(),
            'hd': avg_hd_loss.detach().cpu().item(),
            'init_consistency': init_consistency_loss.detach().cpu().item(),
            'continuity': continuity_loss.detach().cpu().item() if isinstance(continuity_loss, torch.Tensor) else 0.0,
        }
