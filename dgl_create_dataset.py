from dgl_dataset import NASBench101CellDataset
from dgl_params import *
import os


os.makedirs('dgl_dataset/', exist_ok=True)
dataset = NASBench101CellDataset(num_arch=NUM_ARCH, generate_data=True, import_data=False)
