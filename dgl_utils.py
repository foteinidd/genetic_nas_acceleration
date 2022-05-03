import torch

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dgl_dataset import NASBench101CellDataset

from dgl_model import GCN

import numpy as np
from matplotlib import pyplot as plt


def data_preparation(num_arch, train_percentage, batch_size):
    # set seed
    torch.manual_seed(13)

    dataset = NASBench101CellDataset(num_arch=num_arch, generate_data=False)

    num_examples = len(dataset)
    num_train = int(num_examples * train_percentage)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    val_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    val_dataloader = GraphDataLoader(dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)

    return dataset, train_dataloader, val_dataloader


def model_configuration(dataset, learning_rate):
    # set seed
    torch.manual_seed(42)

    model = GCN(dataset.dim_nfeats, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    return model, optimizer, loss_fn


class ModelHandler(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            # set model to train mode
            self.model.train()

            # compute model's predicted output (forward pass)
            yhat = self.model(x)

            # compute loss
            loss = self.loss_fn(yhat, y.view(-1, 1))

            # compute gradients
            loss.backward()

            # update parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # return loss
            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            # set model to evaluation mode
            self.model.eval()

            # compute model's predicted output (forward pass)
            yhat = self.model(x)

            # compute loss
            loss = self.loss_fn(yhat, y.view(-1, 1))

            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)

        for epoch in range(n_epochs):
            self.total_epochs += 1

            # train using minibatches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # validation
            with torch.no_grad():  # no gradients in validation
                # evaluate using minibatches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            print('epoch:', epoch, 'training loss:', loss, 'validation loss:', val_loss)

    def save_checkpoint(self, filename):
        # create checkpoint dictionary
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        # save checkpoint
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # load checkpoint dictionary
        checkpoint = torch.load(filename)

        # restore model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

    def predict(self, x):
        # set model to evaluation mode
        self.model.eval()

        # make prediction
        yhat = self.model(x.to(self.device))

        # set back to training mode
        self.model.train()

        return yhat.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig
