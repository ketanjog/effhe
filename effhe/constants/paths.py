import os
from pathlib import Path

# Base paths for the project.
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")

# MNIST paths
MNIST_PATH = os.path.join(DATA_PATH, "mnist")

# Save path
SAVE_PATH = os.path.join(ROOT_PATH, "checkpoints")
BASELINE_PATH = os.path.join(SAVE_PATH, "mnist_square_1c2f")

