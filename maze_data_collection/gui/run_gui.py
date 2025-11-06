import os, sys
                  
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from fov_renderer import FOVTopViewRenderer, DataCollector

import argparse
from collections import defaultdict

import gym
import numpy as np
import pygame
import pygame.freetype
from gym import spaces
from PIL import Image

from recording import SaveNpzWrapper

from auto_navigator import AutoNavigator

os.environ['MUJOCO_GL'] = 'glfw'

                   
TARGET_COLORS = [
    np.array([170, 38, 30]) / 220,  # red
    np.array([99, 170, 88]) / 220,  # green
    np.array([39, 140, 217]) / 220,  # blue
    np.array([93, 105, 199]) / 220,  # purple
    np.array([220, 193, 59]) / 220,  # yellow
    np.array([220, 128, 107]) / 220,  # salmon
]



         
try:
    from memory_maze.wrappers import VisibilityMaskWrapper
except ImportError:
                       
    class VisibilityMaskWrapper(object):


        def __init__(self, env, visibility_range: float = 3.0, sharing_agents=None):
            self.env = env
            self.visibility_range = visibility_range
            self.sharing_agents = sharing_agents or []

        def __getattr__(self, name):
            return getattr(self.env, name)

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            if isinstance(obs, dict):
                obs = self._apply_visibility_mask(obs)
            return obs, reward, done, info

        def reset(self):
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = self._apply_visibility_mask(obs)
            return obs

        def _generate_visibility_mask(self, agent_pos, maze_size):
            mask = np.zeros((maze_size, maze_size), dtype=np.float32)
            y, x = np.ogrid[:maze_size, :maze_size]
                         
            dist_from_agent = np.sqrt((x - agent_pos[0]) ** 2 + (y - agent_pos[1]) ** 2)
            mask = np.where(dist_from_agent <= self.visibility_range, 1.0, mask)
                          
            for other_agent in self.sharing_agents:
                other_pos = other_agent['position']
                other_range = other_agent.get('visibility_range', self.visibility_range)
                dist_from_other = np.sqrt((x - other_pos[0]) ** 2 + (y - other_pos[1]) ** 2)
                sharing_factor = other_agent.get('sharing_factor', 1.0)
                other_mask = np.where(dist_from_other <= other_range, sharing_factor, 0)
                mask = np.maximum(mask, other_mask)
            return mask

        def _apply_visibility_mask(self, obs):
            if not isinstance(obs, dict) or 'maze_layout' not in obs:
                return obs

            masked_obs = obs.copy()
            maze_size = obs['maze_layout'].shape[0]
            visibility_mask = self._generate_visibility_mask(obs['agent_pos'], maze_size)

                       
            masked_obs['maze_layout'] = obs['maze_layout'] * visibility_mask
            masked_obs['visibility_mask'] = visibility_mask

            if 'targets_pos' in obs:
                visible_targets = []
                visible_indices = []
                for i, target_pos in enumerate(obs['targets_pos']):
                    x, y = target_pos.astype(int)
                    if x < maze_size and y < maze_size and visibility_mask[y, x] > 0:
                        visible_targets.append(target_pos)
                        visible_indices.append(i)
                if visible_targets:
                    masked_obs['targets_pos'] = np.array(visible_targets)
                    masked_obs['visible_target_indices'] = np.array(visible_indices)

            return masked_obs

        def add_sharing_agent(self, position, visibility_range=None, sharing_factor=1.0):
            self.sharing_agents.append({
                'position': np.array(position),
                'visibility_range': visibility_range or self.visibility_range,
                'sharing_factor': sharing_factor
            })

        def clear_sharing_agents(self):
            self.sharing_agents = []

# if 'MUJOCO_GL' not in os.environ:
#     if "linux" in sys.platform:
#         os.environ['MUJOCO_GL'] = 'osmesa' # Software rendering to avoid rendering interference with pygame
#     else:
#         os.environ['MUJOCO_GL'] = 'glfw'  # Windowed rendering

PANEL_LEFT = 250
PANEL_RIGHT = 250
FOCUS_HACK = False
RECORD_DIR = './log'
K_NONE = tuple()


def get_keymap(env):
    return {
        tuple(): 0,
        (pygame.K_UP, ): 1,
        (pygame.K_LEFT, ): 2,
        (pygame.K_RIGHT, ): 3,
        (pygame.K_UP, pygame.K_LEFT): 4,
        (pygame.K_UP, pygame.K_RIGHT): 5,
    }


class TopViewRenderer:
    def __init__(self, size, surface_size=(250, 250)):
        self.size = size
        self.surface_size = surface_size
        self.cell_size = min(surface_size[0] // size, surface_size[1] // size)
        self.surface = pygame.Surface(surface_size)

    def render(self, maze_layout, agent_pos, agent_dir, targets_pos, current_target_idx, visibility_mask=None):
        self.surface.fill((64, 64, 64))

                   
        offset_x = (self.surface_size[0] - self.size * self.cell_size) // 2
        offset_y = (self.surface_size[1] - self.size * self.cell_size) // 2

              
        for y in range(self.size):
            for x in range(self.size):
                if maze_layout[y][x]:
                    pygame.draw.rect(
                        self.surface,
                        (102, 102, 102),
                        (
                            offset_x + x * self.cell_size,
                            offset_y + y * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                    )

              
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
                x, y = pos
                pygame.draw.circle(
                    self.surface,
                    TARGET_COLORS[i % len(TARGET_COLORS)],
                    (
                        offset_x + (x + 0.5) * self.cell_size,
                        offset_y + (y + 0.5) * self.cell_size
                    ),
                    self.cell_size // 4
                )
                if i == current_target_idx:
                    pygame.draw.circle(
                        self.surface,
                        (255, 255, 255),
                        (
                            offset_x + (x + 0.5) * self.cell_size,
                            offset_y + (y + 0.5) * self.cell_size
                        ),
                        self.cell_size // 4,
                        2
                    )

               
        x, y = agent_pos
        angle = np.arctan2(agent_dir[1], agent_dir[0])
        pygame.draw.circle(
            self.surface,
            (255, 215, 0),
            (
                offset_x + (x + 0.5) * self.cell_size,
                offset_y + (y + 0.5) * self.cell_size
            ),
            self.cell_size // 3
        )

                 
        end_pos = (
            offset_x + (x + 0.5) * self.cell_size + np.cos(angle) * (self.cell_size // 3),
            offset_y + (y + 0.5) * self.cell_size + np.sin(angle) * (self.cell_size // 3)
        )
        pygame.draw.line(
            self.surface,
            (0, 0, 0),
            (
                offset_x + (x + 0.5) * self.cell_size,
                offset_y + (y + 0.5) * self.cell_size
            ),
            end_pos,
            2
        )

                    
        if visibility_mask is not None:
            visibility_surface = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
            for y in range(self.size):
                for x in range(self.size):
                    alpha = int((1 - visibility_mask[y][x]) * 200)           
                    pygame.draw.rect(
                        visibility_surface,
                        (0, 0, 0, alpha),
                        (
                            offset_x + x * self.cell_size,
                            offset_y + y * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                    )
            self.surface.blit(visibility_surface, (0, 0))

        return self.surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='memory_maze:MemoryMaze-15x15-v0')
    parser.add_argument('--size', type=int, nargs=2, default=(600, 600))
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for maze generation')
    parser.add_argument('--reset_steps', type=int, default=None,
                        help='Reset maze after this many steps')
    parser.add_argument('--reset_episodes', type=int, default=None,
                        help='Reset maze after this many episodes')
    parser.add_argument('--visibility_range', type=float, default=15.0,
                        help='Agent visibility range in grid cells')
    parser.add_argument('--share_agents', type=str, default=None,
                        help='Sharing agents positions as "x1,y1,r1:x2,y2,r2". Example: "5,5,2:7,7,3"')
    parser.add_argument('--random', type=float, default=0.0)
    parser.add_argument('--noreset', action='store_true')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--nonoop', action='store_true', help='Pause instead of noop')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record_mp4', action='store_true')
    parser.add_argument('--record_gif', action='store_true')
    parser.add_argument('--collect_data', action='store_true',
                        help='Enable data collection mode')
    parser.add_argument('--dataset_dir', type=str, default='bev_predictor/dataset',
                        help='Directory to save collected data')
    parser.add_argument('--fov_angle', type=float, default=90,
                        help='Field of view angle in degrees')
    parser.add_argument('--auto_nav', action='store_true',
                        help='Enable automatic navigation mode')
    parser.add_argument('--max_trajectories', type=int, default=30000,
                        help='Maximum number of trajectories to collect')
    args = parser.parse_args()

    try:
                       
        env = gym.make(args.env, disable_env_checker=True, seed=args.seed)
        
                      
        maze_size = int(args.env.split('-')[1].split('x')[0])
        
                  
        if hasattr(env, 'env') and hasattr(env.env, 'maze_task'):
            maze_arena = env.env.maze_task._maze_arena
            maze_config = {
                'maze_size': maze_size,
                'max_rooms': maze_arena._maze._max_rooms,
                'room_min_size': maze_arena._maze.room_min_size,
                'room_max_size': maze_arena._maze.room_max_size,
                'spawns_per_room': maze_arena._maze.spawns_per_room,
                'objects_per_room': maze_arena._maze.objects_per_room
            }
        else:
                    
            maze_config = {
                'maze_size': maze_size,
                'max_rooms': 6,
                'room_min_size': 3,
                'room_max_size': 5,
                'spawns_per_room': 1,
                'objects_per_room': 1
            }

                                                 
        if args.visibility_range is not None:
            print(f'Applying visibility mask with range: {args.visibility_range}')
            env = VisibilityMaskWrapper(env, visibility_range=args.visibility_range)

        render_size = args.size
        window_size = (render_size[0] + PANEL_LEFT + PANEL_RIGHT, render_size[1])

                             
        if args.share_agents:
            for agent_str in args.share_agents.split(':'):
                if agent_str:
                    x, y, r = map(float, agent_str.split(','))
                    env.add_sharing_agent(
                        position=[x, y],
                        visibility_range=r,
                        sharing_factor=0.8
                    )

        if isinstance(env.observation_space, spaces.Dict):
            print('Observation space:')
            for k, v in env.observation_space.spaces.items():
                print(f'{k:>25}: {v}')
        else:
            print(f'Observation space:  {env.observation_space}')
        print(f'Action space:  {env.action_space}')

        if args.record:
            env = SaveNpzWrapper(
                env,
                RECORD_DIR,
                video_format='mp4' if args.record_mp4 else 'gif' if args.record_gif else None,
                video_fps=args.fps * 2)

               
        maze_size = int(args.env.split('-')[1].split('x')[0])
        regular_top_view = TopViewRenderer(
            size=maze_size,
            surface_size=(PANEL_RIGHT, PANEL_RIGHT)
        )

        fov_top_view = FOVTopViewRenderer(
            size=maze_size,
            surface_size=(PANEL_RIGHT, PANEL_RIGHT),
            fov_angle=args.fov_angle
        )

                  
        if args.auto_nav:
            navigator = AutoNavigator(maze_size=maze_size, fov_angle=args.fov_angle)
            print(f'Automatic navigation enabled. Will collect {args.max_trajectories} trajectories.')

                      
        reset_counter = 0
        data_collector = None
        trajectory_count = 0
        current_dataset_dir = None        

        def init_data_collector(reset_num, current_obs):

            nonlocal current_dataset_dir        
            nonlocal data_collector        
            current_dataset_dir = os.path.join(args.dataset_dir, f'{reset_num:06d}')
            print(f'Data will be saved to: {current_dataset_dir}')

            collector = DataCollector(current_dataset_dir)

                              
            if data_collector:
                data_collector.close()

                       
            data_collector = DataCollector(current_dataset_dir)
            data_collector.initialize_fov_renderer(size=maze_size, fov_angle=args.fov_angle)

                    
            targets_info = []
            for i, pos in enumerate(current_obs['targets_pos']):
                target_info = {
                    'index': i,
                    'position': pos.tolist() if isinstance(pos, np.ndarray) else pos,
                    'color': TARGET_COLORS[i].tolist() if i < len(TARGET_COLORS) else [1.0, 1.0, 1.0]
                }
                targets_info.append(target_info)

                  
            data_collector.save_config(
                env_name=args.env,
                fov_angle=args.fov_angle,
                seed=args.seed,
                maze_size=maze_size,
                maze_config=maze_config,
                targets_info=targets_info
            )
            return data_collector


        keymap = get_keymap(env)

                
        steps = 0
        return_ = 0.0
        episode = 0
        total_steps = 0
        total_episodes = 0
        steps_since_reset = 0
        episodes_since_reset = 0

                             
        obs = env.reset()
        if args.collect_data:
            data_collector = init_data_collector(reset_counter, obs)

        pygame.init()
        start_fullscreen = args.fullscreen or FOCUS_HACK
        screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN if start_fullscreen else 0)
        if FOCUS_HACK and not args.fullscreen:
            pygame.display.toggle_fullscreen()
        clock = pygame.time.Clock()
        font = pygame.freetype.SysFont('Mono', 16)
        fontsmall = pygame.freetype.SysFont('Mono', 12)

        running = True
        paused = False
        speedup = False

        try:
            while running and (not args.auto_nav or trajectory_count < args.max_trajectories):
                # Rendering
                screen.fill((64, 64, 64))

                          
                if isinstance(obs, dict):
                    image = obs['image']
                else:
                    image = obs
                image = Image.fromarray(image)
                image = image.resize(render_size, resample=0)
                image = np.array(image)
                surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
                screen.blit(surface, (PANEL_LEFT, 0))

                        
                lines = obs_to_text(obs, env, steps, return_)
                y = 5
                for line in lines:
                    text_surface, rect = font.render(line, (255, 255, 255))
                    screen.blit(text_surface, (16, y))
                    y += font.size + 2

                        
                lines = keymap_to_text(keymap)
                y = 5
                for line in lines:
                    text_surface, rect = fontsmall.render(line, (255, 255, 255))
                    screen.blit(text_surface, (render_size[0] + PANEL_LEFT + 16, y))
                    y += fontsmall.size + 2

                         
                if isinstance(obs, dict):
                             
                    regular_bev = regular_top_view.render(
                        obs['maze_layout'],
                        obs['agent_pos'],
                        obs['agent_dir'],
                        obs['targets_pos'],
                        obs.get('target_index', 0),
                        obs.get('visibility_mask', None)
                    )
                    screen.blit(regular_bev, (render_size[0] + PANEL_LEFT, 0))

                                 
                    fov_bev = fov_top_view.render(
                        obs['maze_layout'],
                        obs['agent_pos'],
                        obs['agent_dir'],
                        obs['targets_pos'],
                        obs.get('target_index', 0)
                    )
                    screen.blit(fov_bev, (render_size[0] + PANEL_LEFT, PANEL_RIGHT + 10))

                                  
                    if data_collector and not paused:
                        if args.auto_nav:
                            if not navigator.is_stuck():
                                data_collector.save_frame(
                                    obs['image'],
                                    fov_bev,
                                    obs['agent_dir'],
                                    obs                      
                                )
                        else:
                            data_collector.save_frame(
                                obs['image'],
                                fov_bev,
                                obs['agent_dir'],
                                obs                      
                            )
                pygame.display.flip()
                clock.tick(args.fps if not speedup else 0)

                        
                pygame.event.pump()
                keys_down = defaultdict(bool)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        keys_down[event.key] = True
                keys_hold = pygame.key.get_pressed()

                        
                force_reset = False
                speedup = False
                if keys_down[pygame.K_ESCAPE]:
                    running = False
                if keys_down[pygame.K_SPACE]:
                    paused = not paused
                if keys_down[pygame.K_BACKSPACE]:
                    force_reset = True
                if keys_hold[pygame.K_TAB]:
                    speedup = True

                if paused:
                    continue

                              
                action = keymap[K_NONE]

                        
                if args.auto_nav and not keys_down[pygame.K_SPACE]:                 
                    action = navigator.choose_action(obs)
                else:
                    # Action keys
                    for keys, act in keymap.items():
                        if all(keys_hold[key] or keys_down[key] for key in keys):
                            action = act

                if action == keymap[K_NONE] and args.nonoop and not force_reset:
                    continue

                # Environment step
                if args.random:
                    if np.random.random() < args.random:
                        action = env.action_space.sample()

                obs, reward, done, info = env.step(action)
                steps += 1
                total_steps += 1
                steps_since_reset += 1
                return_ += reward

                            
                should_reset = force_reset
                if args.reset_steps and steps_since_reset >= args.reset_steps:
                    should_reset = True
                if args.reset_episodes and episodes_since_reset >= args.reset_episodes:
                    should_reset = True

                                
                if done or should_reset:
                    if args.auto_nav:
                        trajectory_count += 1
                        navigator.reset()           

                    episodes_since_reset += 1
                    total_episodes += 1

                    if should_reset:
                                   
                        if data_collector:
                            data_collector.close()

                                 
                        reset_counter += 1
                        steps_since_reset = 0
                        episodes_since_reset = 0

                              
                        if hasattr(env.env, 'reset_maze'):
                            env.env.reset_maze()

                                           
                        obs = env.reset()
                        if args.collect_data:
                            data_collector = init_data_collector(reset_counter, obs)

                    obs = env.reset()
                    steps = 0
                    return_ = 0.0
                    episode += 1

                    if done and args.record:
                        running = False


        finally:
                           
            if data_collector:
                print(f'Data collection finished.')
                print(f'Total trajectories collected: {trajectory_count}')
                print(f'Total resets: {reset_counter + 1}')
                print(f'Final data directory: {current_dataset_dir}')
                data_collector.close()

            pygame.quit()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if data_collector:
            data_collector.close()
        pygame.quit()
        raise

def obs_to_text(obs, env, steps, return_):
    kvs = []
    kvs.append(('## Stats ##', ''))
    kvs.append(('', ''))
    kvs.append(('step', steps))
    kvs.append(('return', return_))
    lines = [f'{k:<15} {v:>5}' for k, v in kvs]
    return lines


def keymap_to_text(keymap, verbose=False):
    kvs = []
    kvs.append(('## Commands ##', ''))
    kvs.append(('', ''))

    # mapped actions
    kvs.append(('forward', 'up arrow'))
    kvs.append(('left', 'left arrow'))
    kvs.append(('right', 'right arrow'))

    # special actions
    kvs.append(('', ''))
    kvs.append(('reset', 'backspace'))
    kvs.append(('pause', 'space'))
    kvs.append(('speed up', 'tab'))
    kvs.append(('quit', 'esc'))

    lines = [f'{k:<15} {v}' for k, v in kvs]
    return lines


if __name__ == '__main__':
    main()
