import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :, :self.d_model]

class ImprovedEncoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pos_encoder = PositionalEncoding(latent_dim)
        
    def forward(self, x):

        x = self.features(x)  # [B, C, H, W]
        

        B, C, H, W = x.shape
        x = x.permute(3, 0, 2, 1).contiguous()
        x = x.view(W, B*H, C)
        

        x = self.pos_encoder(x)
        x = self.transformer(x)
        

        x = x.view(W, B, H, C).permute(1, 3, 2, 0)
        return x

class ImprovedBEVDecoder(nn.Module):
    def __init__(self, latent_dim=256, target_height=250, target_width=250, latent_h=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_height = target_height 
        self.target_width = target_width
        self.initial_size = 8
        self.latent_h = latent_h
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        self.initial_proj = nn.Linear(self.latent_h * latent_dim, latent_dim)
        

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 4, kernel_size=3, padding=1)
        )

    def forward(self, latent, angle):
        B, C, H, W = latent.shape
        

        latent = latent.permute(3, 0, 2, 1).contiguous()
        latent = latent.reshape(W, B*H, C)
        
        query = torch.zeros(self.initial_size**2, B*H, C, device=latent.device)
        query = self.pos_encoder(query)
        
        output = self.transformer(query, latent)
        

        output = output.permute(1, 2, 0)
        output = output.reshape(B, H * C, self.initial_size, self.initial_size)
        

        output = self.initial_proj(output.permute(0, 2, 3, 1))
        output = output.reshape(B, self.initial_size, self.initial_size, self.latent_dim)
        output = output.permute(0, 3, 1, 2)
        

        x = self.decoder(output)
        rgb = torch.sigmoid(x[:, :3])
        alpha = torch.sigmoid(x[:, 3:])
        

        rgb = F.interpolate(rgb, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        alpha = F.interpolate(alpha, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        
        return torch.cat([rgb, alpha], dim=1)

class BEVPredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = ImprovedEncoder(config.LATENT_DIM)
        self.decoder = ImprovedBEVDecoder(config.LATENT_DIM,
                                        target_height=config.VIZ_OUTPUT_SIZE,
                                        target_width=config.VIZ_OUTPUT_SIZE)

    def forward(self, fps_view, angle):

        latent = self.encoder(fps_view)
        

        bev_pred = self.decoder(latent, angle)

        return {
            'bev': bev_pred,
            'latent': latent
        }
