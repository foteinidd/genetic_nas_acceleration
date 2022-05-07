from dgl_utils import data_preparation, model_configuration, ModelHandler
from dgl_params import *
import numpy as np


dataset, train_dataloader, val_dataloader = data_preparation(num_arch=NUM_ARCH, train_percentage=TRAIN_PERCENTAGE,
                                                             train_batch_size=TRAIN_BATCH_SIZE,
                                                             val_batch_size=VAL_BATCH_SIZE)

model, optimizer, loss_fn = model_configuration(dataset=dataset, num_filters=NUM_FILTERS, learning_rate=LEARNING_RATE)
model_handler = ModelHandler(model=model, optimizer=optimizer, loss_fn=loss_fn)

model_handler.load_checkpoint('dgl_model/dgl_model_checkpoint_epoch40.pth')

for x, y in val_dataloader:
    pred_y = model_handler.predict(x)
    print(pred_y, y, np.abs(pred_y-y.numpy()))
