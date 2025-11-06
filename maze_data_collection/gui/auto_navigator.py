import numpy as np
import random
import math


class AutoNavigator:
    def __init__(self, maze_size=15, base_turn_steps=3, base_forward_steps=2, fov_angle=80):
        # Navigation parameters
        self.ZIGZAG_FORWARD_STEPS = 1  # Steps before turning
        self.ZIGZAG_TURN_ANGLE = 3  # Degrees to turn each time
        self.TURN_STEPS_PER_90_DEGREES = 6  # How many steps needed for 90 degrees
        self.RAY_DIRECTION_THRESHOLD = 90  # Degrees difference to trigger ray-based direction change

        # Convert turn angle to steps
        self.turn_steps = int(self.ZIGZAG_TURN_ANGLE / 90 * self.TURN_STEPS_PER_90_DEGREES)

        # Standard initialization
        self.maze_size = maze_size
        self.visited = np.zeros((maze_size, maze_size), dtype=bool)
        self.zigzag_state = "forward"
        self.steps_taken = 0
        self.turn_direction = "left"
        self.turn_count = 0

        # Keep other initialization parameters
        self.last_pos = None
        self.last_dir = None
        self.stuck_count = 0
        self.consecutive_stuck_positions = []
        self.last_forward_distance = 2
        self.fov_angle = np.deg2rad(fov_angle)
        self.fov_range = 10
        self.base_turn_steps = base_turn_steps
        self.base_forward_steps = base_forward_steps
        self.escape_state = None
        self.escape_steps = 0
        self.required_steps = 0

        self.actions = {
            'forward': 1,
            'turn_left': 2,
            'turn_right': 3
        }

        self.is_currently_stuck = False
        self.displacement_history = []
        self.displacement_window = 3

    def reset(self):
        self.visited = np.zeros((self.maze_size, self.maze_size), dtype=bool)
        self.zigzag_state = "forward"
        self.steps_taken = 0
        self.last_pos = None
        self.last_dir = None
        self.stuck_count = 0
        self.escape_state = None
        self.escape_steps = 0
        self.required_steps = 0
        self.is_currently_stuck = False
        self.displacement_history = []


    def mark_current_visited(self, current_pos):
        y, x = current_pos
        ry, rx = int(round(y)), int(round(x))
        if 0 <= ry < self.maze_size and 0 <= rx < self.maze_size:
            self.visited[ry, rx] = True

                                                    

    def _bresenham_line(self, start, end):

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

    def _is_in_fov(self, target, agent, agent_dir):

        to_target = target - agent
        dist = np.linalg.norm(to_target)
        if dist < 1e-9:
            return True        
        agent_dir_n = agent_dir / (np.linalg.norm(agent_dir) + 1e-9)
        to_target_n = to_target / dist
        angle = np.arccos(np.clip(np.dot(agent_dir_n, to_target_n), -1.0, 1.0))
        return abs(angle) <= self.fov_angle / 2

    def generate_visibility_mask(self, agent_pos, agent_dir, maze_layout):








        size = maze_layout.shape[0]
        visibility = np.zeros((size, size), dtype=bool)

        ax, ay = agent_pos
        ax_int, ay_int = int(ax), int(ay)
        if 0 <= ax_int < size and 0 <= ay_int < size:
            visibility[ay_int, ax_int] = True

        scan_resolution = 120
        max_range = self.fov_range
        agent_angle = math.atan2(agent_dir[1], agent_dir[0])

        for angle_offset in np.linspace(-self.fov_angle/2, self.fov_angle/2, scan_resolution):
            scan_angle = agent_angle + angle_offset
                                
            dist = 0.0
            step = 0.1
            while dist < max_range:
                dist += step
                px = ax + math.cos(scan_angle)*dist
                py = ay + math.sin(scan_angle)*dist
                px_i, py_i = int(round(px)), int(round(py))

                             
                if not (0 <= px_i < size and 0 <= py_i < size):
                    break

                               
                if not self._is_in_fov(np.array([px, py]), np.array([ax, ay]), agent_dir):
                    continue

                                                                
                ray_cells = self._bresenham_line((ax_int, ay_int), (px_i, py_i))
                blocked = False
                for (cx, cy) in ray_cells:
                                    
                    if not (0 <= cx < size and 0 <= cy < size):
                        blocked = True
                        break
                                                             
                                              
                    if maze_layout[cy, cx] == 0:
                        visibility[cy, cx] = True
                        blocked = True
                        break
                    else:
                        visibility[cy, cx] = True

                if blocked:
                    break

        return visibility

                                 

    def check_if_stuck(self, current_pos, agent_dir, maze_layout):
        if self.last_pos is not None and self.last_dir is not None:
            displacement = np.linalg.norm(current_pos - self.last_pos)
            current_angle = math.atan2(agent_dir[1], agent_dir[0])
            last_angle = math.atan2(self.last_dir[1], self.last_dir[0])
            angle_diff = (current_angle - last_angle + math.pi) % (2 * math.pi) - math.pi
            rotation_diff = abs(angle_diff)
            alpha = 0.1
            activity = displacement + alpha * rotation_diff

            self.displacement_history.append(activity)
            if len(self.displacement_history) > self.displacement_window:
                self.displacement_history.pop(0)

            if len(self.displacement_history) == self.displacement_window:
                avg_activity = sum(self.displacement_history) / len(self.displacement_history)
                if avg_activity < 0.05:
                    self.stuck_count += 1
                    if self.stuck_count > 2:
                        wall_distances = self.calculate_wall_distances(current_pos, maze_layout)
                        # print(f"[STUCK] Pos={current_pos}, avg_activity={avg_activity:.4f}")
                        self.is_currently_stuck = True
                        return True, wall_distances
                else:
                    if activity > 0.2:
                        # print("[UNSTUCK] Activity restored!")
                        self.stuck_count = 0
                        self.is_currently_stuck = False
                        self.displacement_history = []

        self.last_pos = current_pos.copy()
        self.last_dir = agent_dir.copy()
        return False, None

    def choose_zigzag_action(self):
        """Implements the zigzag pattern movement"""
        if self.zigzag_state == "forward":
            self.steps_taken += 1
            if self.steps_taken >= self.forward_steps:
                self.zigzag_state = "turning"
                self.steps_taken = 0
                self.turn_count += 1
                self.turn_direction = "right" if self.turn_count % 2 == 0 else "left"
                # print(f"[ZIGZAG] Switching to turning state ({self.turn_direction})")
                return self.actions['turn_right'] if self.turn_direction == "right" else self.actions['turn_left']
            return self.actions['forward']

        elif self.zigzag_state == "turning":
            self.steps_taken += 1
            if self.steps_taken >= self.turn_steps:
                self.zigzag_state = "forward"
                self.steps_taken = 0
                # print("[ZIGZAG] Switching to forward state")
                return self.actions['forward']
            return self.actions['turn_right'] if self.turn_direction == "right" else self.actions['turn_left']

    def choose_action(self, obs):
        current_pos = obs['agent_pos']
        agent_dir = obs['agent_dir']
        maze_layout = obs['maze_layout']

        self.mark_current_visited(current_pos)

        if self.escape_state:
            return self.choose_action_in_escape(self.forward_steps_for_escape)

        is_stuck, wall_distances = self.check_if_stuck(current_pos, agent_dir, maze_layout)
        if is_stuck:
            self.forward_steps_for_escape = self.start_escape_sequence(wall_distances, current_pos, agent_dir,
                                                                       maze_layout)
            return self.actions['turn_right'] if self.turn_direction == "right" else self.actions['turn_left']

        # Ray detection for direction optimization
        best_angle = self.find_best_direction_by_mask(current_pos, agent_dir, maze_layout)
        if best_angle is not None:
            current_angle = math.atan2(agent_dir[1], agent_dir[0])
            angle_diff = (best_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
            angle_diff_degrees = math.degrees(abs(angle_diff))

            if angle_diff_degrees > self.RAY_DIRECTION_THRESHOLD:
                # Bias the zigzag pattern towards the detected direction
                self.turn_direction = "right" if angle_diff > 0 else "left"
                # print(f"[RAY] Biasing towards {self.turn_direction}, angle_diff={angle_diff_degrees:.1f} deg")
                return self.actions[f'turn_{self.turn_direction}']

        # Regular zigzag pattern
        if self.zigzag_state == "forward":
            self.steps_taken += 1
            if self.steps_taken >= self.ZIGZAG_FORWARD_STEPS:
                self.zigzag_state = "turning"
                self.steps_taken = 0
                self.turn_count += 1
                self.turn_direction = "right" if self.turn_count % 2 == 0 else "left"
                return self.actions[f'turn_{self.turn_direction}']
            return self.actions['forward']

        elif self.zigzag_state == "turning":
            self.steps_taken += 1
            if self.steps_taken >= self.turn_steps:
                self.zigzag_state = "forward"
                self.steps_taken = 0
                return self.actions['forward']
            return self.actions[f'turn_{self.turn_direction}']

    def calculate_wall_distances(self, current_pos, maze_layout):
               
        y, x = map(int, current_pos)
        distances = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

        # up
        for i in range(y-1, -1, -1):
            if maze_layout[i, x] == 0:
                distances['up'] = y - i
                break
            if i == 0:
                distances['up'] = y + 1

        # down
        for i in range(y+1, self.maze_size):
            if maze_layout[i, x] == 0:
                distances['down'] = i - y
                break
            if i == self.maze_size-1:
                distances['down'] = self.maze_size - y

        # left
        for i in range(x-1, -1, -1):
            if maze_layout[y, i] == 0:
                distances['left'] = x - i
                break
            if i == 0:
                distances['left'] = x + 1

        # right
        for i in range(x+1, self.maze_size):
            if maze_layout[y, i] == 0:
                distances['right'] = i - x
                break
            if i == self.maze_size-1:
                distances['right'] = self.maze_size - x

        return distances

    def is_stuck(self):
        return self.is_currently_stuck

    def start_escape_sequence(self, wall_distances, current_pos, agent_dir, maze_layout):
                            
        turn_steps = random.randint(4, 8)
        forward_steps = 3
        if random.random() < 0.5:
            self.turn_direction = self.actions['turn_left']
        else:
            self.turn_direction = self.actions['turn_right']

        self.escape_state = 'turning'
        self.escape_steps = 0
        self.required_steps = turn_steps

        # print(f"[Escape] Spin {turn_steps} steps, then {forward_steps} forward.")
        return forward_steps

    def choose_action_in_escape(self, forward_steps):
        if self.escape_state == 'turning':
            self.escape_steps += 1
            if self.escape_steps >= self.required_steps:
                self.escape_state = 'forward'
                self.escape_steps = 0
                self.required_steps = forward_steps
                # print("[Escape] Turn finished, start forward.")
                return self.actions['forward']
            return self.turn_direction
        elif self.escape_state == 'forward':
            self.escape_steps += 1
            if self.escape_steps >= self.required_steps:
                self.escape_state = None
                self.escape_steps = 0
                # print("[Escape] Escape done.")
            return self.actions['forward']
        return self.actions['forward']

                                                         
    def find_best_direction_by_mask(self, current_pos, agent_dir, maze_layout):







        visibility = self.generate_visibility_mask(current_pos, agent_dir, maze_layout)

                       
        unvisited_coords = []
        ay, ax = current_pos
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                if visibility[y, x] and not self.visited[y, x]:
                    unvisited_coords.append((x, y))

        if not unvisited_coords:
            return None         

                                
                                        
        angles = []
        for (ux, uy) in unvisited_coords:
            dx = ux - ax
            dy = uy - ay
            angle = math.atan2(dy, dx)
            angles.append(angle)

                           
        vx = sum(math.cos(a) for a in angles)
        vy = sum(math.sin(a) for a in angles)
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            return None
        best_angle = math.atan2(vy, vx)
        return best_angle
