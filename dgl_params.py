NUM_ARCH = 500
TRAIN_PERCENTAGE = 0.3
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
NUM_FILTERS = 150
LEARNING_RATE = 0.001
DROPOUT_PROBABILITY = 0
NUM_EPOCHS = 100
TRAIN_STATS_SAVE_DIR = 'dgl_model_train_stats'
EVAL_STATS_SAVE_DIR = 'dgl_model_evaluation_stats'
MODEL_NAME = 'gcn_{}filters_5sageconv_lstm_scaled_relu_trainperc{}_{}epochs_{}nasbench101archs'.format(NUM_FILTERS,
                                                                                                       # int(DROPOUT_PROBABILITY*100),
                                                                                                       int(TRAIN_PERCENTAGE * 100),
                                                                                                       NUM_EPOCHS,
                                                                                                       NUM_ARCH)
