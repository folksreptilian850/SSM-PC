import os
from pathlib import Path


class SPCConfig:
    """Configuration for social place-cell experiments."""

    _DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
    DATA_ROOT = os.environ.get("SPC_DATA_ROOT", str(_DEFAULT_DATA_ROOT))
    DATASET_PATTERN = os.environ.get("SPC_DATASET_PATTERN", "D*")

    ENV_SIZE = 15
    PLACE_CELLS_N = 256
    PLACE_CELLS_SCALE = 0.5
    HD_CELLS_N = 12
    HD_CELLS_CONCENTRATION = 20

    HIDDEN_SIZE = 128
    DROPOUT_RATE = 0.5
    LATENT_DIM = 256
    SEQUENCE_LENGTH = 100
    SEQUENCE_STRIDE = 1
    SEED = 42

    @classmethod
    def to_dict(cls):
        return {
            "ENV_SIZE": cls.ENV_SIZE,
            "PLACE_CELLS_N": cls.PLACE_CELLS_N,
            "PLACE_CELLS_SCALE": cls.PLACE_CELLS_SCALE,
            "HD_CELLS_N": cls.HD_CELLS_N,
            "HD_CELLS_CONCENTRATION": cls.HD_CELLS_CONCENTRATION,
            "HIDDEN_SIZE": cls.HIDDEN_SIZE,
            "DROPOUT_RATE": cls.DROPOUT_RATE,
            "LATENT_DIM": cls.LATENT_DIM,
            "SEQUENCE_LENGTH": cls.SEQUENCE_LENGTH,
            "SEQUENCE_STRIDE": cls.SEQUENCE_STRIDE,
            "SEED": cls.SEED,
        }
