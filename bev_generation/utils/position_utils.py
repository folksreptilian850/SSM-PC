# /bev_predictor/utils/position_utils.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

class PositionConverter:
    def __init__(self, maze_size=15):
        self.maze_size = maze_size

    def absolute_to_relative(self, obj_pos, agent_pos, agent_angle):












        dx = obj_pos[0] - agent_pos[0]
        dy = obj_pos[1] - agent_pos[1]
        

        rot_angle = -(agent_angle - np.pi/2)
        rot_matrix = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)]
        ])
        
        rel_pos = np.dot(rot_matrix, [dx, dy])


        rel_pos[0] = -rel_pos[0]
        

        rel_pos = rel_pos / (self.maze_size / 2)
        
        return rel_pos

    def relative_to_absolute(self, relative_pos, agent_pos, agent_angle):












        relative_pos = relative_pos * (self.maze_size / 2)
        

        rot_angle = agent_angle - np.pi/2
        rot_matrix = np.array([
            [np.cos(-rot_angle), -np.sin(-rot_angle)],
            [np.sin(-rot_angle), np.cos(-rot_angle)]
        ])
        
        world_offset = np.dot(rot_matrix, relative_pos)
        

        absolute_pos = np.array(agent_pos) + world_offset
        
        return absolute_pos
        
def relative_to_absolute_torch(relative_pos, agent_pos, agent_angle, maze_size):









    B = relative_pos.shape[0]
    N = relative_pos.shape[1] if relative_pos.dim() == 3 else 1


    scale = maze_size / 2.0
    relative_meters = relative_pos * scale  # [B, N, 2]


    rot_angle = agent_angle - (torch.pi / 2)

    rot_angle = rot_angle.view(-1)

    cos_ = torch.cos(rot_angle)  # [B]
    sin_ = torch.sin(rot_angle)  # [B]


    x_prime = relative_meters[..., 0]  # [B, N]
    y_prime = relative_meters[..., 1]  # [B, N]




    X = cos_.unsqueeze(-1) * x_prime - sin_.unsqueeze(-1) * y_prime
    Y = sin_.unsqueeze(-1) * x_prime + cos_.unsqueeze(-1) * y_prime


    # agent_pos: [B, 2]
    Ax = agent_pos[:, 0].unsqueeze(-1)  # [B,1]
    Ay = agent_pos[:, 1].unsqueeze(-1)  # [B,1]
    X = X + Ax
    Y = Y + Ay


    absolute_pos = torch.stack([X, Y], dim=-1)  # [B, N, 2]

    return absolute_pos

def visualize_positions(bev_image, relative_positions, visible_indices):








    if isinstance(bev_image, Image.Image):
        bev_array = np.array(bev_image)
    else:
        bev_array = bev_image
        
    height, width = bev_array.shape[:2]
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_array)
    

    plt.plot(width/2, height/2, 'r+', markersize=10)
    

    colors = plt.cm.rainbow(np.linspace(0, 1, len(relative_positions)))
    for i, (dx, dy) in enumerate(relative_positions):
        if i in visible_indices:

            x = width/2 + dx * width/2
            y = height/2 + dy * height/2
            
            plt.plot(x, y, 'o', color=colors[i], markersize=8, label=f'Object {i}')
            

            plt.plot([width/2, x], [height/2, y], '--', color=colors[i], alpha=0.5)
    
    plt.legend()
    plt.title('BEV with Object Positions')
    plt.axis('off')
    return plt.gcf()

if __name__ == "__main__":

    converter = PositionConverter(maze_size=15)
    

    agent_pos = [7.5, 7.5]
    agent_angle = np.pi/2
    obj_pos = [7.5, 10.5]
    
    relative_pos = converter.absolute_to_relative(obj_pos, agent_pos, agent_angle)
    print(f"Test 1 - Object in front:")
    print(f"Relative position: {relative_pos}")
    
    recovered_pos = converter.relative_to_absolute(relative_pos, agent_pos, agent_angle)
    print(f"Recovered absolute position: {recovered_pos}")
    print(f"Original absolute position: {obj_pos}")
    print(f"Conversion error: {np.linalg.norm(np.array(obj_pos) - recovered_pos)}")
    

    obj_pos2 = [10.5, 7.5]
    relative_pos2 = converter.absolute_to_relative(obj_pos2, agent_pos, agent_angle)
    print(f"\nTest 2 - Object on right:")
    print(f"Relative position: {relative_pos2}")
    

    bev_image = np.ones((250, 250, 4), dtype=np.uint8) * 255
    relative_positions = [relative_pos, relative_pos2]
    visible_indices = [0, 1]
    fig = visualize_positions(bev_image, relative_positions, visible_indices)
    plt.show()
