from dgl_dataset import NASBench101CellDataset
import os


os.makedirs('dgl_dataset/', exist_ok=True)
dataset = NASBench101CellDataset(num_arch=10000, generate_data=True)
