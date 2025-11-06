import os
from pathlib import Path


class BEVConfig:
    def __init__(self):
              
        self.INPUT_CHANNELS = 3         
        self.INPUT_HEIGHT = 64
        self.INPUT_WIDTH = 64
        self.BEV_HEIGHT = 250
        self.BEV_WIDTH = 250
        self.BEV_CHANNELS = 4  # RGB + Alpha
        
              
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 100
        
                
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]

class Config:
           
    _DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "subset"
    DATA_ROOT = os.environ.get("BEV_DATA_ROOT", str(_DEFAULT_DATA_ROOT))
    DATASET_PATTERN = os.environ.get("BEV_DATASET_PATTERN", "D*")                            
    TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]                  

    @property
    def DATASET_DIRS(self):

        import glob
        import os
        
                         
        dirs = glob.glob(os.path.join(self.DATA_ROOT, self.DATASET_PATTERN))
                     
        dirs = [d for d in dirs if os.path.isdir(d)]
                       
        dirs.sort()
                           
        return [os.path.basename(d) for d in dirs]
    
          
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LEARNING_RATE = 3e-4
    SAVE_FREQUENCY = 5                          
    
            
    LATENT_DIM = 256          
    MAX_OBJECTS = 6           
    
            
    TARGET_RADIUS = 0.6           
    
            
    LOSS_WEIGHTS = {
                 
        'alpha_weight': 1.0,                   
        'rgb_weight': 0.5,                   
        'smoothness_weight': 0.1,          
        
                
        'visibility_weight': 1.0,           
        'position_weight': 1.0,            
    }
    
          
    USE_MIXED_PRECISION = True              
    NUM_WORKERS = 4                        
    PIN_MEMORY = True                               
    
          
    EVAL_FREQUENCY = 1                          
    EVAL_SAMPLES = 2                      
    VISIBILITY_THRESHOLD = 0.5           
    
           
    SAVE_VISUALIZATIONS = True             
    VIZ_OUTPUT_SIZE = 250                  
    
                    
    CHECKPOINT_DIR = None          
    INFERENCE_DIR = None             
    
               
    POST_PROCESS = {
        'nms_threshold': 0.3,                
        'confidence_threshold': 0.5,         
        'min_distance': 0.1,                  
    }

              
    VISUAL_DIM = 256                       
    GRID_CELL_DIM = 100                      
    GRID_SCALES = [1.0, 2.0, 4.0]                  
    
              
    NAV_LOSS_WEIGHTS = {
        'position': 1.0,            
        'code': 0.1                 
    }
    
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

class TestConfig(Config):

    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.USE_MIXED_PRECISION = False

class DebugConfig(Config):

    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 2
        self.NUM_EPOCHS = 2
        self.SAVE_FREQUENCY = 1
        self.NUM_WORKERS = 0

if __name__ == "__main__":
          
    config = Config()
    
               
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmp_dir:
              
        save_path = os.path.join(tmp_dir, 'config.json')
        config.save(save_path)
        
              
        loaded_config = Config.load(save_path)
        
                 
        assert loaded_config.BATCH_SIZE == config.BATCH_SIZE
        assert loaded_config.MAX_OBJECTS == config.MAX_OBJECTS
        print("配置测试通过！")
    
             
    train_config = TrainConfig()
    test_config = TestConfig()
    debug_config = DebugConfig()
    
    print(f"训练批大小: {train_config.BATCH_SIZE}")
    print(f"测试批大小: {test_config.BATCH_SIZE}")
    print(f"调试批大小: {debug_config.BATCH_SIZE}")
