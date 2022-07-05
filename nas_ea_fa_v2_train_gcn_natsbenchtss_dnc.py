from nord.neural_nets.natsbench_evaluator import NATSBench_Evaluator

import os
import copy
import time
import numpy as np

from params import EXP_REPEAT_TIMES, MAX_TIME_BUDGET, POPULATION_SIZE, NUM_GEN, K, H, T
from natsbenchtss_utils_dnc import Architecture, build_connection_matrix
from natsbenchtss_utils_dnc import randomly_sample_architecture, create_nord_architecture, \
    get_all_isomorphic_sequences, get_min_distance, get_model_sequences, tournament_selection, bitwise_mutation

from performance_evaluation import progress_update, save_performance
from save_individual import save_individual_101_dnc, save_individual_fitness_approximation

import torch
import dgl
from dgl_utils import data_preparation, model_configuration, ModelHandler
from dgl_params import *


def NAS_EA_FA_V2_train_gcn_201():
    # Instantiate the evaluator
    evaluator = NATSBench_Evaluator()

    dir_name_results = 'results_nas_ea_fa_v2_dnc201_train_gcn_{}epochs_trainperc{}_K{}_H{}_populationsize{}'.format(NUM_EPOCHS,
                                                                                                                            int(TRAIN_PERCENTAGE * 100),
                                                                                                                            K, H, POPULATION_SIZE)
    dir_name_dgl_train_stats = 'nas_ea_fa_v2_dnc201_dgl_model_train_stats_{}epochs_trainperc{}_K{}_H{}_populationsize{}'.format(NUM_EPOCHS,
                                                                                                                                        int(TRAIN_PERCENTAGE * 100),
                                                                                                                                        K, H, POPULATION_SIZE)
    dir_name_dgl_eval_stats = 'nas_ea_fa_v2_dnc201_dgl_model_eval_stats_{}epochs_trainperc{}_K{}_H{}_populationsize{}'.format(NUM_EPOCHS,
                                                                                                                                      int(TRAIN_PERCENTAGE * 100),
                                                                                                                                      K, H, POPULATION_SIZE)

    if not os.path.exists(dir_name_results):
        os.mkdir(dir_name_results)
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = os.path.join(dir_name_results, 'results' + str(exp_repeat_index + 1))
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_val_acc = []
        best_test_acc_based_on_val_acc = []
        train_times = []
        total_train_time = []

        best_test_acc = []

        x_train = []
        x_train_graphs = []
        y_train = []

        current_time_budget = 0

        # Randomly sample POPULATION_SIZE architectures with an initial fitness of 0
        total_population = []
        for _ in range(POPULATION_SIZE):
            is_valid_architecture = False
            while not is_valid_architecture:
                architecture = randomly_sample_architecture()
                if architecture.valid_architecture:

                    d = create_nord_architecture(architecture)

                    # evaluate architecture
                    invalid_nas201 = False
                    try:
                        val_acc, test_acc, train_time = evaluator.descriptor_evaluate(d, metrics=['validation_accuracy',
                                                                                                  'test_accuracy',
                                                                                                  'time_cost'])
                    except ValueError:
                        # print('Invalid architecture (not added in population)')
                        # print(d)
                        # print(evaluator._descriptor_to_nasnet(d))
                        invalid_nas201 = True

                    if not invalid_nas201:
                        total_population.append(architecture)
                        is_valid_architecture = True

        num_file = 0

        t = 0  # iteration count
        # while current_time_budget <= MAX_TIME_BUDGET:
        while t < T:
            tic = time.time()
            t += 1
            # sort in descending order by fitness
            population = sorted(total_population, key=lambda x: x.fitness, reverse=True)
            new_population = []
            num_arch = 0
            start_index = 0

            # train and evaluate top K individuals
            for arch_index in range(len(population)):
                architecture = population[arch_index]

                d = create_nord_architecture(architecture)

                # evaluate architecture
                train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, latency, train_time = \
                    evaluator.descriptor_evaluate(d, metrics=['train_loss',
                                                              'validation_loss',
                                                              'test_loss',
                                                              'train_accuracy',
                                                              'validation_accuracy',
                                                              'test_accuracy',
                                                              'latency',
                                                              'time_cost'])

                architecture.fitness = val_acc
                architecture.test_acc = test_acc
                architecture.train_time = train_time

                architecture.train_loss = train_loss
                architecture.val_loss = val_loss
                architecture.test_loss = test_loss
                architecture.train_acc = train_acc
                architecture.latency = latency

                if time == 0.0:
                    continue

                new_population.append(architecture)

                # get isomorphic sequences
                isomorphic_sequences = get_all_isomorphic_sequences(architecture)
                for sequence in isomorphic_sequences:
                    a = Architecture(architecture.layers, sequence, build_connection_matrix(sequence),
                                     architecture.fitness)
                    if a.valid_architecture:
                        d = create_nord_architecture(a)
                        # convert neural descriptor to networkx
                        G = d.to_networkx(benchmark_dataset='natsbenchtss')
                        # convert networkx to dgl object
                        a.dgl_obj = dgl.from_networkx(G, node_attrs=['layer'])
                        x_train.append(sequence)
                        x_train_graphs.append(a.dgl_obj)
                        y_train.append(val_acc*0.01)
                # x_train.extend(isomorphic_sequences)
                # for _ in range(len(isomorphic_sequences)):
                #     y_train.append(val_acc)

                best_val_acc, best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time = \
                    progress_update(val_acc=val_acc, test_acc=test_acc, train_time=train_time,
                                    best_val_acc=best_val_acc,
                                    best_test_acc_based_on_val_acc=best_test_acc_based_on_val_acc,
                                    best_test_acc=best_test_acc, train_times=train_times,
                                    total_train_time=total_train_time, fitness='val_acc')

                current_time_budget += train_time

                num_arch += 1

                if current_time_budget > MAX_TIME_BUDGET or num_arch >= K:
                    start_index = arch_index
                    break

            num_file += 1
            with open(os.path.join(folder_name, 'topK_iteration' + str(num_file) + '.txt'), 'w') as f:
                ind_num = 0
                for ind in new_population:
                    ind_num += 1
                    save_individual_101_dnc(f, ind, ind_num, 'val_acc')

            num_topK = len(new_population)

            # train and evaluate top H individuals
            tic1 = time.time()
            # get min distance between each of the remaining individuals and the training set
            dist_list = [get_min_distance(x_train, get_model_sequences(architecture)) for architecture in
                         population[start_index + 1:]]
            toc1 = time.time()
            print('x_train length:', len(x_train))
            print('dist_list calculation time:', toc1 - tic1, 'sec')

            while num_arch < K + H and current_time_budget <= MAX_TIME_BUDGET:
                # find architecture with max distance from training set
                max_distance = 0
                max_dist_arch_index = start_index
                for i in range(len(dist_list)):
                    if dist_list[i] > max_distance:
                        max_distance = dist_list[i]
                        max_dist_arch_index = i

                architecture = population[start_index + 1 + max_dist_arch_index]
                dist_list[max_dist_arch_index] = 0  # architecture already added to x_train

                d = create_nord_architecture(architecture)

                # evaluate architecture
                train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, latency, train_time = \
                    evaluator.descriptor_evaluate(d, metrics=['train_loss',
                                                              'validation_loss',
                                                              'test_loss',
                                                              'train_accuracy',
                                                              'validation_accuracy',
                                                              'test_accuracy',
                                                              'latency',
                                                              'time_cost'])

                architecture.fitness = val_acc
                architecture.test_acc = test_acc
                architecture.train_time = train_time

                architecture.train_loss = train_loss
                architecture.val_loss = val_loss
                architecture.test_loss = test_loss
                architecture.train_acc = train_acc
                architecture.latency = latency

                if time == 0.0:
                    continue

                new_population.append(architecture)

                # get isomorphic sequences
                isomorphic_sequences = get_all_isomorphic_sequences(architecture)
                for sequence in isomorphic_sequences:
                    a = Architecture(architecture.layers, sequence, build_connection_matrix(sequence),
                                     architecture.fitness)
                    if a.valid_architecture:
                        d = create_nord_architecture(a)
                        # convert neural descriptor to networkx
                        G = d.to_networkx(benchmark_dataset='natsbenchtss')
                        # convert networkx to dgl object
                        a.dgl_obj = dgl.from_networkx(G, node_attrs=['layer'])
                        x_train.append(sequence)
                        x_train_graphs.append(a.dgl_obj)
                        y_train.append(val_acc*0.01)
                # x_train.extend(isomorphic_sequences)
                # for _ in range(len(isomorphic_sequences)):
                #     y_train.append(val_acc)

                best_val_acc, best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time = \
                    progress_update(val_acc=val_acc, test_acc=test_acc, train_time=train_time,
                                    best_val_acc=best_val_acc,
                                    best_test_acc_based_on_val_acc=best_test_acc_based_on_val_acc,
                                    best_test_acc=best_test_acc, train_times=train_times,
                                    total_train_time=total_train_time, fitness='val_acc')

                current_time_budget += train_time

                num_arch += 1

            with open(os.path.join(folder_name, 'topH_iteration' + str(num_file) + '.txt'), 'w') as f:
                ind_num = num_topK
                for index in range(num_topK, len(new_population)):
                    ind = new_population[index]
                    ind_num += 1
                    save_individual_101_dnc(f, ind, ind_num, 'val_acc')

            # update population
            if len(new_population) != 0:
                population = new_population

            # # train fitness approximation
            # with open(os.path.join(folder_name, 'xgb_stats_iteration' + str(num_file) + '.txt'), 'w') as f:
            #     with redirect_stdout(f):
            #         # xgb_model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1)
            #         xgb_model = XGBRegressor(eta=0.1)
            #         if t > 1:
            #             xgb_model.fit(np.array(x_train), np.array(y_train), eval_set=[(x_train, y_train), (x_val, y_val)],
            #                           eval_metric='rmse')
            #         else:
            #             xgb_model.fit(np.array(x_train), np.array(y_train), eval_set=[(x_train, y_train)],
            #                           eval_metric='rmse')
            #         xgb_stats = xgb_model.evals_result()
            #         print(xgb_stats)

            # data setup
            dataset, train_data_loader, val_data_loader = data_preparation(train_percentage=TRAIN_PERCENTAGE,
                                                                           train_batch_size=TRAIN_BATCH_SIZE,
                                                                           val_batch_size=VAL_BATCH_SIZE,
                                                                           num_arch=len(x_train_graphs),
                                                                           graphs=x_train_graphs,
                                                                           labels=torch.Tensor(y_train))
            # dataset.labels = dataset.labels - 0.5
            # model setup
            model, optimizer, loss_fn = model_configuration(dataset=dataset, num_filters=NUM_FILTERS,
                                                            learning_rate=LEARNING_RATE,
                                                            dropout_probability=DROPOUT_PROBABILITY)
            model_handler = ModelHandler(model=model, optimizer=optimizer, loss_fn=loss_fn)
            model_handler.set_loaders(train_loader=train_data_loader, val_loader=val_data_loader)
            # model_handler.to('cpu')
            model_handler.train(n_epochs=NUM_EPOCHS,
                                stats_save_dir=os.path.join(dir_name_dgl_train_stats,
                                                            'experiment' + str(exp_repeat_index + 1)),
                                model_name='iteration' + str(t))

            model_handler.plot_losses(
                stats_save_dir=os.path.join(dir_name_dgl_train_stats, 'experiment' + str(exp_repeat_index + 1)),
                model_name='iteration' + str(t))

            # model_handler.load_checkpoint(os.path.join('dgl_model', 'gcn_150filters_5sageconv_lstm_scaled_relu_trainperc30_100epochs_500nasbench101archs_checkpoint_epoch{}.pth'.format(NUM_EPOCHS)))

            model_handler.evaluate(
                stats_save_dir=os.path.join(dir_name_dgl_eval_stats, 'experiment' + str(exp_repeat_index + 1)),
                model_name='iteration' + str(t))

            with open(os.path.join(folder_name, 'len_data' + str(num_file) + '.txt'), 'w') as f:
                f.write(str(len(x_train_graphs)) + '\n')

            # evolutionary algorithm
            total_population = []
            for epoch in range(NUM_GEN):
                new_population = []
                for i in range(POPULATION_SIZE):
                    individual = copy.deepcopy(tournament_selection(population))
                    new_individual = bitwise_mutation(individual)

                    while not new_individual.valid_architecture:
                        individual = copy.deepcopy(tournament_selection(population))
                        new_individual = bitwise_mutation(individual)

                    d = create_nord_architecture(new_individual)
                    # convert neural descriptor to networkx
                    G = d.to_networkx(benchmark_dataset='natsbenchtss')
                    # convert networkx to dgl object
                    new_individual.dgl_obj = dgl.from_networkx(G, node_attrs=['layer'])

                    # new_individual.fitness = xgb_model.predict(np.array([get_model_sequences(new_individual)]))[0]
                    new_individual.fitness = model_handler.predict(new_individual.dgl_obj)[0][0]  # + 0.5

                    new_population.append(new_individual)
                    total_population.append(new_individual)

                population = new_population

                with open(os.path.join(folder_name, 'population_iteration' + str(num_file) + '_epoch' + str(epoch + 1) +
                                                    '.txt'), 'w') as f:
                    ind_num = 0
                    for ind in population:
                        ind_num += 1
                        save_individual_fitness_approximation(f, ind, ind_num, 'val_acc')

            # # validation set for next iteration's xgboost model
            # x_val = x_train
            # y_val = y_train

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for iteration t=' + str(t) + ':', toc - tic,
                  'sec')
            print('current time budget:', current_time_budget, 'max time budget:', MAX_TIME_BUDGET)

        end_time = time.time()

        save_performance(folder_name, exp_repeat_index, start_time, end_time, best_val_acc,
                         best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time,
                         'val_acc')


if __name__ == '__main__':
    np.random.seed(42)
    NAS_EA_FA_V2_train_gcn_201()
