from dgl_utils import data_preparation, model_configuration, ModelHandler

from matplotlib import pyplot as plt
import os


# data setup
dataset, train_dataloader, val_dataloader = data_preparation(num_arch=10000, train_percentage=0.8, batch_size=128)

# model setup
model, optimizer, loss_fn = model_configuration(dataset=dataset, learning_rate=0.01)

# train
model_handler = ModelHandler(model=model, optimizer=optimizer, loss_fn=loss_fn)
model_handler.set_loaders(train_loader=train_dataloader, val_loader=val_dataloader)
model_handler.train(n_epochs=100)

print(model.state_dict())
model_handler.plot_losses()
plt.show()

os.makedirs('dgl_model/', exist_ok=True)
model_handler.save_checkpoint('dgl_model/dgl_model_checkpoint.pth')
