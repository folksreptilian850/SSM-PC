import os
from pathlib import Path


class Config:
    """
    Default configuration for the grid-cell training and analysis pipeline.

    Paths default to the repository layout: place datasets inside
    ``grid_cell/data/self-navigation-maze-frame-only`` or override with
    environment variables.
    """

    _DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "self-navigation-maze-frame-only"
    DATA_ROOT = os.environ.get("GRID_DATA_ROOT", str(_DEFAULT_DATA_ROOT))
    DATASET_PATTERN = os.environ.get("GRID_DATASET_PATTERN", "D*_P*")
    TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]
    SEQUENCE_LENGTH = 100
    SEQUENCE_STRIDE = 1

    @property
    def DATASET_DIRS(self):

        import glob
        import os

        dirs = glob.glob(os.path.join(self.DATA_ROOT, self.DATASET_PATTERN))
        dirs = [d for d in dirs if os.path.isdir(d)]
        dirs.sort()
        return [os.path.basename(d) for d in dirs]

                         
    ENV_SIZE = 15           
    PLACE_CELLS_N = 256                 
    PLACE_CELLS_SCALE = 0.5                    
    HD_CELLS_N = 12                          
    HD_CELLS_CONCENTRATION = 20                 

    HIDDEN_SIZE = 128              
    DROPOUT_RATE = 0.5             

        
    SEED = 42        
    RESUME = None              

          
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    # LEARNING_RATE = 5e-6

                                       
    NATURE_LEARNING_RATE = 1e-3
    NATURE_WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1

    SAVE_FREQUENCY = 5

            
    LATENT_DIM = 256
    MAX_OBJECTS = 6
    HIDDEN_DIM = 256
    MAZE_SIZE = 15

              
    MEMORY_SIZE = 10000         
    MEMORY_TEMPERATURE = 0.1                  
    MEMORY_SPARSITY_WEIGHT = 0.01                
    MEMORY_ENTROPY_WEIGHT = 0.01              
    MEMORY_UPDATE_FREQ = 10                    
    MEMORY_TOP_K = 3                

            
    TARGET_RADIUS = 0.6

                       
    LOSS_WEIGHTS = {
        'alpha_weight': 1.0,
        'rgb_weight': 0.5,
        'smoothness_weight': 0.1,
        'visibility_weight': 1.0,
        'position_weight': 1.0,
    }

    NAV_LOSS_WEIGHTS = {
        'position': 1.0,  # Keep position loss dominant
        'rotation': 10,  # Increase rotation weight slightly
        'consistency': 0.05,  # Keep low to encourage smoothness
        'memory_sparsity': 0.005,  # Reduce sparsity loss weight
        'memory_entropy': 0.005,  # Reduce entropy loss weight slightly
        'grid_periodicity': 0.0,
        'grid_sparsity': 0.0,
        'grid_consistency': 0.0
    }

          
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 4
    PIN_MEMORY = True

          
    EVAL_FREQUENCY = 1
    EVAL_SAMPLES = 2
    VISIBILITY_THRESHOLD = 0.5

           
    SAVE_VISUALIZATIONS = False
    VIZ_OUTPUT_SIZE = 250
    VIZ_INTERVAL = 20
    VIZ_MAX_SAMPLES = 8
    VIZ_TYPES = [
        'predicted_positions',
        'place_memory_matches',               
        'bev_output'
    ]

          
    CHECKPOINT_DIR = None
    INFERENCE_DIR = None

                 
    BEV_PRETRAINED_PATH = os.environ.get("GRID_BEV_PRETRAINED_PATH")
    TARGET_PRETRAINED_PATH = os.environ.get("GRID_TARGET_PRETRAINED_PATH")

                               
            
    DATA_CHUNKING_ENABLED = True          
    DATA_CHUNK_SIZE = 1000            
    ROTATE_CHUNKS_WITHIN_EPOCH = True                    
    SHUFFLE_CHUNKS = True              

            
    LAZY_LOADING_ENABLED = True                  
    FRAME_INFO_CACHE_SIZE = 100            
    PRELOAD_METADATA_ONLY = True                    

            
    MEMORY_MANAGEMENT_ENABLED = True          
    MEMORY_THRESHOLD_MB = 10000                      
    FORCE_GC_BETWEEN_CHUNKS = True               
    CACHE_CLEARING_FREQUENCY = 10               

             
    WORKER_MEMORY_LIMIT_MB = 2048                   
    WORKER_TIMEOUT = 60             
    PREFETCH_FACTOR = 1                   
    MAX_ACTIVE_WORKERS = 2             

                  
    GRADIENT_ACCUMULATION_STEPS = 1          
    ADAPTIVE_BATCH_SIZE = False             
    MIN_BATCH_SIZE = 1          

           
    MEMORY_PROFILING_ENABLED = True            
    LOG_MEMORY_USAGE = True            
    MEMORY_LOG_FREQUENCY = 10                

            
    CHECKPOINT_CHUNK_STATE = True              
    RESUME_CHUNK_TRAINING = True               

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    @staticmethod
    def from_dict(config_dict):
        return Config(**config_dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('__') and not callable(v)}

    def save(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


                        
class TrainConfig(Config):
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 32
        self.USE_MIXED_PRECISION = True
                         
        self.DATA_CHUNK_SIZE = 1000
        self.WORKER_MEMORY_LIMIT_MB = 4096               
        self.GRADIENT_ACCUMULATION_STEPS = 2                 


class TestConfig(Config):
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.USE_MIXED_PRECISION = False
                       
        self.DATA_CHUNK_SIZE = 250             
        self.LAZY_LOADING_ENABLED = True
        self.NUM_WORKERS = 1                  


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 2
        self.NUM_EPOCHS = 2
        self.SAVE_FREQUENCY = 1
        self.NUM_WORKERS = 0
                         
        self.MEMORY_PROFILING_ENABLED = True
        self.LOG_MEMORY_USAGE = True
        self.MEMORY_LOG_FREQUENCY = 1             
        self.DATA_CHUNK_SIZE = 100               


                
class LargeDatasetConfig(Config):
    def __init__(self):
        super().__init__()
                    
        self.BATCH_SIZE = 32
        self.DATA_CHUNK_SIZE = 100
        self.LAZY_LOADING_ENABLED = True
        self.PRELOAD_METADATA_ONLY = True
        self.MEMORY_MANAGEMENT_ENABLED = True
        self.FORCE_GC_BETWEEN_CHUNKS = True
        self.WORKER_MEMORY_LIMIT_MB = 1024
        self.NUM_WORKERS = 10
        self.PREFETCH_FACTOR = 1
        self.GRADIENT_ACCUMULATION_STEPS = 4
        self.PIN_MEMORY = False
        self.LOG_MEMORY_USAGE = True

