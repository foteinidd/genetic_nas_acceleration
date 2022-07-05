from dgl_utils import data_preparation, model_configuration, ModelHandler
from dgl_params import *

from torchinfo import summary


# data setup
dataset, train_data_loader, val_data_loader = data_preparation(num_arch=NUM_ARCH, train_percentage=TRAIN_PERCENTAGE,
                                                               train_batch_size=TRAIN_BATCH_SIZE,
                                                               val_batch_size=VAL_BATCH_SIZE)
# dataset.labels = dataset.labels - 0.5

# model setup
model, optimizer, loss_fn = model_configuration(dataset=dataset, num_filters=NUM_FILTERS, learning_rate=LEARNING_RATE,
                                                dropout_probability=DROPOUT_PROBABILITY)

# train
model_handler = ModelHandler(model=model, optimizer=optimizer, loss_fn=loss_fn)

# print model summary
summary(model)
# print model
print(model)

model_handler.set_loaders(train_loader=train_data_loader, val_loader=val_data_loader)
# model_handler.to('cpu')
model_handler.train(n_epochs=NUM_EPOCHS, stats_save_dir=TRAIN_STATS_SAVE_DIR, model_name=MODEL_NAME, chkpt_save_dir='dgl_model')

# print(model.state_dict())
model_handler.plot_losses(stats_save_dir=TRAIN_STATS_SAVE_DIR, model_name=MODEL_NAME)
# plt.show()

model_handler.evaluate(stats_save_dir=EVAL_STATS_SAVE_DIR, model_name=MODEL_NAME)
