import dgl
from dgl.data import DGLDataset

import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import torch
import numpy as np

from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, MAX_CONNECTIONS
from nord.neural_nets import BenchmarkEvaluator


class NASBench101CellDataset(DGLDataset):
    def __init__(self, num_arch, generate_data=False, import_data=False, graphs=None, labels=None):
        self.num_arch = num_arch  # number of cells in dataset
        self.generate_data = generate_data
        self.import_data = import_data
        if self.import_data:
            self.graphs = graphs
            self.labels = labels
            if len(self.graphs[0].ndata['layer'].shape) == 1:
                self.dim_nfeats = 1
            else:
                self.dim_nfeats = self.graphs[0].ndata['layer'].shape[1]
        else:
            self.graphs = []
            self.labels = []
            self.dim_nfeats = 0
        super().__init__(name='nasbench101_cell', save_dir='dgl_dataset/')

    def process(self):
        if not self.import_data:
            if self.generate_data:
                evaluator = BenchmarkEvaluator()

                # set random seed
                np.random.seed(seed=42)

                for i in range(self.num_arch):
                    # generate random architecture
                    is_valid_architecture = False
                    while not is_valid_architecture:
                        architecture = randomly_sample_architecture()
                        # check if connection number is ok for nasbench-101
                        if sum(architecture.connections) <= MAX_CONNECTIONS and architecture.valid_architecture:
                            is_valid_architecture = True

                    # build neural descriptor object
                    d = create_nord_architecture(architecture)

                    val_acc, train_time = evaluator.descriptor_evaluate(d, acc='validation_accuracy')

                    # convert neural descriptor to networkx
                    G = d.to_networkx()

                    # convert networkx to dgl object
                    dgl_obj = dgl.from_networkx(G, node_attrs=['layer'])

                    self.graphs.append(dgl_obj)
                    self.labels.append(val_acc)

                if len(self.graphs[0].ndata['layer'].shape) == 1:
                    self.dim_nfeats = 1
                else:
                    self.dim_nfeats = self.graphs[0].ndata['layer'].shape[1]

                self.save()
            else:
                self.load()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_dir + 'dgl_graph.bin')
        save_graphs(graph_path, list(self.graphs), {'fitness': torch.Tensor(np.array(self.labels))})
        info_dict = {'dim_nfeats': self.dim_nfeats}
        save_info(os.path.join(self.save_dir + 'dgl_graph_info.pkl'), info_dict)

    def load(self):
        # load processed data from directory `self.save_dir`
        graph_path = os.path.join(self.save_dir + 'dgl_graph.bin')
        self.graphs, fitness_dict = load_graphs(graph_path)
        self.labels = fitness_dict['fitness']
        info_dict = load_info(os.path.join(self.save_dir + 'dgl_graph_info.pkl'))
        self.dim_nfeats = info_dict['dim_nfeats']


if __name__ == '__main__':
    dataset = NASBench101CellDataset(num_arch=10000)
    print(dataset)
    print(dataset.__len__())
    graph, fitness = dataset[0]
    print(graph, fitness)
