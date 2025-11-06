# datasets/navigation_dataset.py
import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import json
import numpy as np



class SingleMazeDataset(Dataset):


    def __init__(self, maze_dir, sequence_length=4, stride=1, lazy_loading=True):
        super().__init__()
        self.maze_dir = maze_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.lazy_loading = lazy_loading


        with open(os.path.join(maze_dir, 'frame_info.json'), 'r') as f:

            if lazy_loading:

                frame_info_temp = json.load(f)
                self.frame_info = {}
                for k, v in frame_info_temp.items():
                    self.frame_info[k] = {
                        'frame_id': v.get('frame_id'),
                        'timestamp': v.get('timestamp')
                    }
            else:
                self.frame_info = json.load(f)


        self.sequences = self._create_sequences()


        if lazy_loading and hasattr(self, 'frame_info_full'):
            del self.frame_info_full


    def __len__(self):

        if not hasattr(self, 'sequences') or self.sequences is None:
            return 0
        return len(self.sequences)

    def _split_sequence(self, sequence):

        splits = []
        for i in range(0, len(sequence) - self.sequence_length + 1, self.stride):
            splits.append(sequence[i:i + self.sequence_length])
        return splits

    def _create_sequences(self):

        sequences = []


        sorted_frames = sorted(self.frame_info.items(),
                               key=lambda x: int(x[1]['frame_id']))


        current_sequence = []
        prev_frame_num = None

        for frame_id, info in sorted_frames:
            frame_num = int(info['frame_id'])

            if prev_frame_num is None or frame_num == prev_frame_num + 1:
                current_sequence.append((frame_id, info))
            else:
                if len(current_sequence) >= self.sequence_length:
                    sequences.extend(self._split_sequence(current_sequence))
                current_sequence = [(frame_id, info)]

            prev_frame_num = frame_num


        if len(current_sequence) >= self.sequence_length:
            sequences.extend(self._split_sequence(current_sequence))

        return sequences

    def _load_full_frame_info(self, frame_id):

        if not self.lazy_loading or 'agent_state' in self.frame_info[frame_id]:
            return self.frame_info[frame_id]


        frame_path = os.path.join(self.maze_dir, f'frame_info_{frame_id}.json')
        if os.path.exists(frame_path):
            try:
                with open(frame_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass


        if not hasattr(self, 'frame_info_full'):
            with open(os.path.join(self.maze_dir, 'frame_info.json'), 'r') as f:
                self.frame_info_full = json.load(f)

        return self.frame_info_full.get(frame_id, {})

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        batch_data = {
            'angles': [],
            'positions': [],
            'velocities': [],
            'angular_velocities': [],
            'timestamps': [],
            'frame_ids': []
        }


        dataset_dir = os.path.basename(os.path.dirname(self.maze_dir))

        for frame_id, info in sequence:

            if self.lazy_loading:
                info = self._load_full_frame_info(frame_id)


            agent_state = info.get('agent_state', {})
            position = agent_state.get('position', [0, 0, 0])
            rotation = agent_state.get('rotation', 0)
            velocity = agent_state.get('velocity', [0, 0, 0])
            angular_velocity = agent_state.get('angular_velocity', 0)

            batch_data['positions'].append(np.array(position, dtype=np.float32))
            batch_data['angles'].append(np.float32(rotation))
            batch_data['velocities'].append(np.array(velocity, dtype=np.float32))
            batch_data['angular_velocities'].append(np.float32(angular_velocity))
            batch_data['timestamps'].append(np.float32(info.get('timestamp', 0)))

            batch_data['frame_ids'].append(frame_id.replace('frame_', ''))


        tensor_keys = ['positions', 'angles', 'velocities',
                       'angular_velocities', 'timestamps']
        for key in tensor_keys:
            if key in batch_data and batch_data[key]:
                batch_data[key] = torch.stack([
                    x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
                    for x in batch_data[key]
                ])


        batch_data['sequence_id'] = dataset_dir

        return batch_data


# 1. Add data chunking mechanism to EnhancedNavigationDataset class
class EnhancedNavigationDataset(Dataset):


    def __init__(self, maze_dirs, sequence_length=4, stride=1, split='train', chunk_size=1000, current_chunk=0):
        super().__init__()
        self.maze_dirs = maze_dirs
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split


        self.chunk_size = chunk_size
        self.current_chunk = current_chunk
        self.total_chunks = None


        self._load_chunk()



    def _load_chunk(self):

        self.datasets = []


        if self.total_chunks is None:


            total_dirs = len(self.maze_dirs)
            self.total_chunks = max(1, (total_dirs + self.chunk_size - 1) // self.chunk_size)


        self.current_chunk = min(max(0, self.current_chunk), self.total_chunks - 1)


        start_idx = self.current_chunk * self.chunk_size
        end_idx = min((self.current_chunk + 1) * self.chunk_size, len(self.maze_dirs))

        chunk_maze_dirs = self.maze_dirs[start_idx:end_idx]

        for maze_dir in chunk_maze_dirs:
            try:
                dataset = SingleMazeDataset(maze_dir, self.sequence_length, self.stride)
                if len(dataset) > 0:
                    self.datasets.append(dataset)
            except Exception as e:
                print(f"Warning: Failed to load dataset from {maze_dir}: {e}")

        if not self.datasets:
            print(f"Warning: No valid datasets found in chunk {self.current_chunk}!")

            try:

                self.datasets = [SingleMazeDataset(self.maze_dirs[0], self.sequence_length, self.stride)]
            except Exception as e:

                print(f"无法创建备用数据集: {e}")
                from torch.utils.data import TensorDataset
                dummy_data = torch.zeros((1, self.sequence_length, 3))
                self.datasets = [TensorDataset(dummy_data)]


        self._calculate_cumulative_lengths()

    def switch_chunk(self, new_chunk_index):

        if new_chunk_index < 0 or (self.total_chunks is not None and new_chunk_index >= self.total_chunks):
            print(f"Warning: Chunk index {new_chunk_index} out of valid range [0, {self.total_chunks-1 if self.total_chunks else '?'}]")
            return False
        
        if new_chunk_index == self.current_chunk:
            return True
        
        try:
            old_chunk = self.current_chunk
            self.current_chunk = new_chunk_index
            self._load_chunk()
            return True
        except Exception as e:
            print(f"切换数据块失败: {e}")
            self.current_chunk = old_chunk
            return False

    def _calculate_cumulative_lengths(self):

        self.cumulative_lengths = []
        current_length = 0
        for dataset in self.datasets:
            current_length += len(dataset)
            self.cumulative_lengths.append(current_length)

    def __len__(self):
        if not self.cumulative_lengths:
            return 0
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):

        if idx < 0 or not self.cumulative_lengths or idx >= self.cumulative_lengths[-1]:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
        

        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) and idx >= self.cumulative_lengths[dataset_idx]:
            dataset_idx += 1
        

        if dataset_idx >= len(self.datasets):
            raise IndexError(f"Dataset index {dataset_idx} out of range (max: {len(self.datasets)-1})")
        

        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        

        if local_idx < 0 or local_idx >= len(self.datasets[dataset_idx]):
            raise IndexError(f"Local index {local_idx} out of range for dataset {dataset_idx}")
        
        try:
            return self.datasets[dataset_idx][local_idx]
        except Exception as e:
            print(f"Error accessing dataset {dataset_idx}, local_idx {local_idx}")
            print(f"Datasets length: {len(self.datasets)}, Cumulative lengths: {self.cumulative_lengths}")
            raise
