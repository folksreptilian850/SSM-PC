import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SpatialFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
                        
        resnet = resnet18(pretrained=True)
                       
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )             
            
                                            
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

    def forward(self, x):
                  
        features = self.backbone(x)  # [B, 256, H/8, W/8]
        
                
        spatial_features = self.spatial_conv(features)
        
        return spatial_features

class PositionPredictor(nn.Module):
    def __init__(self, max_objects=6):
        super().__init__()
        self.max_objects = max_objects
        
                                    
        self.shared_attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, max_objects, 1)                 
        )
        self.softmax = nn.Softmax(dim=2)
        
                  
        self.position_decoder = nn.Sequential(
            nn.Linear(64 + 2, 64),           
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
    def forward(self, features, angle):
        B, C, H, W = features.shape
        
                     
        attn_logits = self.shared_attention(features)  # [B, max_objects, H, W]
        attention_maps = attn_logits.view(B, self.max_objects, -1)
        attention_maps = self.softmax(attention_maps)
        attention_maps = attention_maps.view(B, self.max_objects, H, W)
        
                   
        positions = []
        for i in range(self.max_objects):
                  
            attn = attention_maps[:, i:i+1]  # [B, 1, H, W]
            weighted_features = features * attn
            pooled = torch.mean(weighted_features.view(B, C, -1), dim=2)
            
                        
            features_with_angle = torch.cat([pooled, angle], dim=1)
            pos = self.position_decoder(features_with_angle)
            positions.append(pos)
        
        positions = torch.stack(positions, dim=1)  # [B, max_objects, 2]
        
        return positions, attention_maps

class TargetDetector(nn.Module):
    def __init__(self, max_objects=6):
        super().__init__()
        self.max_objects = max_objects
        
               
        self.feature_extractor = SpatialFeatureExtractor()
        
               
        self.position_predictor = PositionPredictor(max_objects)
        
                         
        self.visibility_predictor = nn.Sequential(
            nn.Linear(64 + 2, 64),                         
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, max_objects),
            nn.Sigmoid()
        )

    def forward(self, fps_view, angle):
              
        spatial_features = self.feature_extractor(fps_view)
        
               
        global_features = torch.mean(spatial_features, dim=(2, 3))
        features_with_angle = torch.cat([global_features, angle], dim=1)
        visibility = self.visibility_predictor(features_with_angle)
        
              
        positions, attention_maps = self.position_predictor(spatial_features, angle)
        
                   
        positions = positions * visibility.unsqueeze(-1)
        
        return {
            'attention_maps': attention_maps,
            'object_visibility': visibility,
            'object_relative_positions': positions
        }

            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

      
if __name__ == "__main__":
          
    model = TargetDetector(max_objects=6)
    
            
    total_params = count_parameters(model)
    print(f"Model has {total_params:,} parameters")
    
            
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 64, 64)
    angle_tensor = torch.randn(batch_size, 2)
    
    outputs = model(input_tensor, angle_tensor)
    
            
    print(f"Attention maps shape: {outputs['attention_maps'].shape}")
    print(f"Visibility shape: {outputs['object_visibility'].shape}")
    print(f"Positions shape: {outputs['object_relative_positions'].shape}")
