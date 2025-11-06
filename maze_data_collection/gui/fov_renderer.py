import numpy as np
import pygame
from PIL import Image
import os
from datetime import datetime
import json


class FOVTopViewRenderer:
    def __init__(self, size, surface_size=(250, 250), fov_angle=90):
        self.size = size
        self.surface_size = surface_size
        self.cell_size = min(surface_size[0] // size, surface_size[1] // size)

                                          
        self.actual_surface_size = (
            self.cell_size * size,
            self.cell_size * size
        )
                 
        self.center_offset_x = (surface_size[0] - self.actual_surface_size[0]) // 2
        self.center_offset_y = (surface_size[1] - self.actual_surface_size[1]) // 2

        self.surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        self.fov_angle = np.deg2rad(fov_angle)
        self.current_agent_dir = None

    def _generate_visibility_mask(self, agent_pos, agent_dir, maze_layout):

        visibility = np.zeros((self.size, self.size), dtype=bool)
        ax, ay = agent_pos
        agent_cell = (int(ax), int(ay))

                        
        ix, iy = agent_cell
        if 0 <= ix < self.size and 0 <= iy < self.size:
            visibility[iy, ix] = True

                
        scan_resolution = 180           
        max_range = 10
        step_size = 0.2

                            
        base_angle = np.arctan2(agent_dir[1], agent_dir[0])

                      
        for theta in np.linspace(-self.fov_angle / 2, self.fov_angle / 2, scan_resolution):
            current_angle = base_angle + theta
            scan_dir = np.array([np.cos(current_angle), np.sin(current_angle)])

                      
            for dist in np.arange(step_size, max_range + step_size, step_size):
                point = agent_pos + scan_dir * dist
                px, py = int(point[0]), int(point[1])

                             
                if not (0 <= px < self.size and 0 <= py < self.size):
                    break

                            
                if not self._is_in_fov((px, py), agent_pos, agent_dir):
                    continue

                                 
                ray_cells = self._bresenham_line(agent_cell, (px, py))
                blocked = False

                for rx, ry in ray_cells[1:]:
                    if not (0 <= rx < self.size and 0 <= ry < self.size):
                        blocked = True
                        break

                    if not maze_layout[ry, rx]:       
                        visibility[ry, rx] = True          
                        blocked = True
                        break
                    visibility[ry, rx] = True           

                if blocked:
                    break

        return visibility

    def _bresenham_line(self, start, end):
        """Bresenham's line algorithm for grid line drawing"""
        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]), int(end[1])

        cells = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1

        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        cells.append((x, y))
        return cells

    def _is_in_fov(self, target_pos, agent_pos, agent_dir):

        if np.all(target_pos == agent_pos):
            return True

        to_target = np.array([target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1]])
        if np.all(to_target == 0):
            return True

        to_target_norm = to_target / np.linalg.norm(to_target)
        agent_dir_norm = agent_dir / np.linalg.norm(agent_dir)

                          
        angle = np.arccos(np.clip(np.dot(to_target_norm, agent_dir_norm), -1.0, 1.0))

                                   
        if angle > np.pi / 2:
            return False

                       
        return abs(angle) <= self.fov_angle / 2

    def check_visibility(self, target_pos, agent_pos, agent_dir, maze_layout):

                     
        if not self._is_in_fov(target_pos, agent_pos, agent_dir):
            return False

                      
        ray_cells = self._bresenham_line(agent_pos, target_pos)
        for x, y in ray_cells[1:]:        
            if not (0 <= x < self.size and 0 <= y < self.size):
                return False
            if not maze_layout[y, x]:        
                return False
        return True

    def get_visible_targets(self, targets_pos, agent_pos, agent_dir, maze_layout):

        visibility_mask = self._generate_visibility_mask(agent_pos, agent_dir, maze_layout)
        visible_indices = []

        for i, target_pos in enumerate(targets_pos):
            x, y = int(target_pos[0]), int(target_pos[1])
            if 0 <= x < self.size and 0 <= y < self.size and visibility_mask[y, x]:
                visible_indices.append(i)

        return visible_indices if visible_indices else [-1]

    def render(self, maze_layout, agent_pos, agent_dir, targets_pos, current_target_idx):

        self.surface.fill((0, 0, 0, 0))          
        self.current_agent_dir = agent_dir

                 
        visibility = self._generate_visibility_mask(agent_pos, agent_dir, maze_layout)

                                    
                                
        target_agent_x = self.surface_size[0] // 2 + self.cell_size // 2
        target_agent_y = self.surface_size[1] // 2 + self.cell_size // 2

        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])

                
        visible_range = 10
        for dy in range(-visible_range, visible_range + 1):
            for dx in range(-visible_range, visible_range + 1):
                x = agent_x + dx
                y = agent_y + dy

                if 0 <= x < self.size and 0 <= y < self.size and visibility[y, x]:
                                            
                    screen_x = target_agent_x + dx * self.cell_size - self.cell_size // 2
                    screen_y = target_agent_y + dy * self.cell_size - self.cell_size // 2

                    if maze_layout[y][x]:      
                        pygame.draw.rect(
                            self.surface,
                            (102, 102, 102, 255),
                            (screen_x, screen_y, self.cell_size, self.cell_size)
                        )
                                 
                    #     pygame.draw.rect(
                    #         self.surface,
                    #         (64, 64, 64, 255),
                    #         (screen_x, screen_y, self.cell_size, self.cell_size)
                    #     )

                 
        TARGET_COLORS = [
            (170, 38, 30),  # red
            (99, 170, 88),  # green
            (39, 140, 217),  # blue
            (93, 105, 199),  # purple
            (220, 193, 59),  # yellow
            (220, 128, 107),  # salmon
        ]

        if isinstance(targets_pos, np.ndarray) and len(targets_pos) > 0:
            for i, pos in enumerate(targets_pos):
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < self.size and 0 <= y < self.size and visibility[y, x]:
                    dx = x - agent_x
                    dy = y - agent_y
                    screen_x = target_agent_x + dx * self.cell_size
                    screen_y = target_agent_y + dy * self.cell_size

                    color = list(TARGET_COLORS[i % len(TARGET_COLORS)]) + [255]
                    pygame.draw.circle(
                        self.surface,
                        color,
                        (screen_x, screen_y),
                        self.cell_size // 4
                    )
                    if i == current_target_idx:
                        pygame.draw.circle(
                            self.surface,
                            (255, 255, 255, 255),
                            (screen_x, screen_y),
                            self.cell_size // 4,
                            2
                        )

                  
        pygame.draw.circle(
            self.surface,
            (255, 215, 0, 255),
            (target_agent_x, target_agent_y),
            self.cell_size // 3
        )

                 
        angle = np.arctan2(agent_dir[1], agent_dir[0])
        end_pos = (
            target_agent_x + np.cos(angle) * (self.cell_size // 2),
            target_agent_y + np.sin(angle) * (self.cell_size // 2)
        )
        pygame.draw.line(
            self.surface,
            (0, 0, 0, 255),
            (target_agent_x, target_agent_y),
            end_pos,
            2
        )

        return self.surface

def is_wall_collision(image, center_ratio=0.5, threshold=13, border_thickness=5):

    h, w = image.shape[:2]
    inner_image = image[border_thickness:h-border_thickness, border_thickness:w-border_thickness]
    h, w = inner_image.shape[:2]
    center_h_start = int(h * (1 - center_ratio) / 2)
    center_h_end = int(h * (1 + center_ratio) / 2)
    center_w_start = int(w * (1 - center_ratio) / 2)
    center_w_end = int(w * (1 + center_ratio) / 2)
    
    center_region = inner_image[center_h_start:center_h_end, center_w_start:center_w_end]
    top_region = inner_image[:center_h_start, :]
    bottom_region = inner_image[center_h_end:, :]
    left_region = inner_image[:, :center_w_start]
    right_region = inner_image[:, center_w_end:]
    
    center_mean = np.mean(center_region, axis=(0, 1))
    top_mean = np.mean(top_region, axis=(0, 1))
    bottom_mean = np.mean(bottom_region, axis=(0, 1))
    left_mean = np.mean(left_region, axis=(0, 1))
    right_mean = np.mean(right_region, axis=(0, 1))
    
    edges_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4
    color_diff = np.linalg.norm(center_mean - edges_mean)
    
    return color_diff < threshold


class DataCollector:
    def __init__(self, output_dir="dataset", buffer_size=3):
        self.output_dir = output_dir
        self.fps_dir = os.path.join(output_dir, "fps_views")
        self.bev_dir = os.path.join(output_dir, "bev_views")
        os.makedirs(self.fps_dir, exist_ok=True)
        os.makedirs(self.bev_dir, exist_ok=True)
        self.frame_count = 0
        self.info_dir = output_dir
        self.frame_info = {}
        self.buffer = []
        self.buffer_size = buffer_size
        self.fov_renderer = None
        self.targets_info = None

    def _convert_to_native_types(self, obj):

        if isinstance(obj, np.ndarray):
            return self._convert_to_native_types(obj.tolist())
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return obj


    def initialize_fov_renderer(self, size, fov_angle):

        self.fov_renderer = FOVTopViewRenderer(size=size, fov_angle=fov_angle)

    def save_frame(self, fps_view, bev_view, agent_dir, obs_dict, is_stuck=False):

                
        wall_collision = is_wall_collision(fps_view)
        is_stuck = is_stuck or wall_collision

                
        visible_indices = [-1]             
        try:
            if self.fov_renderer and all(k in obs_dict for k in ['maze_layout', 'agent_pos', 'targets_pos']):
                visible_indices = self.fov_renderer.get_visible_targets(
                    obs_dict['targets_pos'],
                    obs_dict['agent_pos'],
                    obs_dict['agent_dir'],
                    obs_dict['maze_layout']
                )
        except Exception as e:
            print(f"Warning: Error checking object visibility: {str(e)}")

        frame_data = {
            'fps_view': fps_view.copy(),
            'bev_view': bev_view.copy(),
            'agent_dir': agent_dir.copy(),
            'frame_id': self.frame_count,
            'visible_indices': visible_indices,
            'obs_dict': obs_dict                         
        }

        self.buffer.append(frame_data)

        if is_stuck:
            self.buffer.clear()
        elif len(self.buffer) > self.buffer_size:
            oldest_frame = self.buffer.pop(0)
            self._save_single_frame(**oldest_frame)

        self.frame_count += 1

    def _save_single_frame(self, fps_view, bev_view, agent_dir, frame_id, visible_indices, obs_dict):

                 
        fps_img = Image.fromarray(fps_view)
        fps_img.save(os.path.join(self.fps_dir, f"frame_{frame_id:06d}.png"))

                 
        bev_str = pygame.image.tostring(bev_view, 'RGBA')
        bev_img = Image.frombytes('RGBA', bev_view.get_size(), bev_str)
        bev_img.save(os.path.join(self.bev_dir, f"frame_{frame_id:06d}.png"))

                     
        angle = float(np.arctan2(agent_dir[1], agent_dir[0]))
        
                   
        agent_pos = obs_dict['agent_pos'].tolist() if isinstance(obs_dict['agent_pos'], np.ndarray) else obs_dict['agent_pos']
        
                           
        self.frame_info[f"frame_{frame_id:06d}"] = {
            "rotation_angle": angle,
            "visible_indices": visible_indices,
            "agent_position": agent_pos                
        }

        if frame_id % 100 == 0:
            self._save_frame_info()

    def _save_frame_info(self):

                           
        frame_info = self._convert_to_native_types(self.frame_info)
        frame_info_path = os.path.join(self.info_dir, "frame_info.json")
        with open(frame_info_path, 'w') as f:
            json.dump(frame_info, f, indent=4)

    def save_config(self, env_name, fov_angle, seed, maze_size, maze_config, targets_info):

                           
        config = {
            "env_name": str(env_name),
            "agent_height": 0.3,
            "fov_angle": float(fov_angle),
            "seed": None if seed is None else int(seed),
            "maze_size": int(maze_size),
            "maze_config": self._convert_to_native_types(maze_config),
            "objects": self._convert_to_native_types(targets_info),
            "created_at": datetime.now().isoformat()
        }

        config_path = os.path.join(self.info_dir, "env_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

                          
        self.targets_info = self._convert_to_native_types(targets_info)

    def close(self):

        for frame in self.buffer:
            self._save_single_frame(**frame)
        self.buffer.clear()
        self._save_frame_info()
