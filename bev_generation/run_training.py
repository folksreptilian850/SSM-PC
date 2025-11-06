# /bev_predictor/run_training.py

import os
import sys

                  
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from train import train_model
from config import Config

if __name__ == '__main__':
    config = Config()
    train_model(config)
