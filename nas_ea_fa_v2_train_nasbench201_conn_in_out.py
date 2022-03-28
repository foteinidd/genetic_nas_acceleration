import numpy as np
from nord.neural_nets.natsbench_evaluator import NATSBench_Evaluator
import os
from xgboost import XGBRegressor
import copy
from contextlib import redirect_stdout
import time
import traceback

from nasbench201_utils_cio import EXP_REPEAT_TIMES, MAX_TIME_BUDGET, POPULATION_SIZE, NUM_GEN, K, H
from nasbench201_utils_cio import randomly_sample_architecture, create_nord_architecture, \
    get_all_isomorphic_sequences, get_min_distance, get_model_sequences, tournament_selection, bitwise_mutation


def NAS_EA_FA_V2():
    # Instantiate the evaluator
    evaluator = NATSBench_Evaluator()

    if not os.path.exists('results_nas_ea_fa_v2_cio201_train'):
        os.mkdir('results_nas_ea_fa_v2_cio201_train')
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = 'results_nas_ea_fa_v2_cio201_train/results' + str(exp_repeat_index + 1)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_val_acc = []
        best_test_acc = []
        train_times = []
        total_time = []

        x_train = []
        y_train = []

        current_time_budget = 0

        # Randomly sample POPULATION_SIZE architectures with an initial fitness of 0
        total_population = []
        for _ in range(POPULATION_SIZE):
            is_valid_architecture = False
            while not is_valid_architecture:
                architecture = randomly_sample_architecture()

                d = create_nord_architecture(architecture)

                # evaluate architecture
                invalid_nas201 = False
                try:
                    val_acc, test_acc, train_time = evaluator.descriptor_evaluate(d, metrics=['validation_accuracy',
                                                                                              'test_accuracy', 'time_cost'])
                except ValueError:
                    print('Invalid architecture (not added in population)')
                    print(d)
                    print(evaluator._descriptor_to_nasnet(d))
                    invalid_nas201 = True
                except:
                    print('Exception found')
                    print(traceback.format_exc())
                    print(d)
                    invalid_nas201 = True

                if not invalid_nas201:
                    total_population.append(architecture)
                    is_valid_architecture = True

        num_file = 0

        t = 0  # iteration count
        while current_time_budget <= MAX_TIME_BUDGET:
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
                val_acc, test_acc, train_time = evaluator.descriptor_evaluate(d, metrics=['validation_accuracy',
                                                                                          'test_accuracy', 'time_cost'])

                architecture.fitness = val_acc

                architecture.test_acc = test_acc
                architecture.train_time = train_time

                if time == 0.0:
                    continue

                new_population.append(architecture)
                # x_train.append(get_sequence(get_node_encoding(architecture.layers).flatten(), architecture.connections))
                # y_train.append(val_acc)

                # print('architecture:', architecture.layers)

                # get isomorphic sequences
                isomorphic_sequences = get_all_isomorphic_sequences(architecture)
                x_train.extend(isomorphic_sequences)
                for _ in range(len(isomorphic_sequences)):
                    y_train.append(val_acc)
                # # print('isomorphic sequences (topK) initial length:', len(isomorphic_sequences))
                # isomorphic_sequences_unique = []
                # for arr in isomorphic_sequences:
                #   if not any(np.array_equal(arr, unique_arr) for unique_arr in isomorphic_sequences_unique):
                #     isomorphic_sequences_unique.append(arr)
                # # print('isomorphic sequences (topK) length:', len(isomorphic_sequences_unique))
                # x_train.extend(isomorphic_sequences_unique)
                # for _ in range(len(isomorphic_sequences_unique)):
                #     y_train.append(val_acc)

                if best_val_acc != []:
                    if val_acc > best_val_acc[-1]:
                        best_val_acc.append(val_acc)
                        best_test_acc.append(test_acc)
                    else:
                        best_val_acc.append(best_val_acc[-1])
                        best_test_acc.append(best_test_acc[-1])
                else:
                    best_val_acc.append(val_acc)
                    best_test_acc.append(test_acc)
                train_times.append(train_time)
                if total_time != []:
                    total_time.append(total_time[-1] + train_time)
                else:
                    total_time.append(train_time)
                current_time_budget += train_time

                num_arch += 1

                if current_time_budget > MAX_TIME_BUDGET or num_arch >= K:
                    start_index = arch_index
                    break

            num_file += 1
            with open(folder_name + '/topK_iteration' + str(num_file) + '.txt', 'w') as f:
                ind_num = 0
                for ind in new_population:
                    ind_num += 1
                    f.write('architecture' + str(ind_num) + '\n')
                    f.write('layers: ')
                    for op in ind.layers:
                        f.write(op + ' ')
                    f.write('\n')
                    f.write('connections: ')
                    for conn in ind.connections:
                        f.write(str(int(conn)) + ' ')
                    f.write('\n')
                    f.write('fitness (validation accuracy): ')
                    f.write(str(ind.fitness))
                    f.write('\n')
                    f.write('test accuracy: ')
                    f.write(str(ind.test_acc))
                    f.write('\n')
                    f.write('train time: ')
                    f.write(str(ind.train_time))
                    f.write('\n')

            num_topK = len(new_population)

            # print('experiment index:', exp_repeat_index, 't=' + str(t), 'got to line 109')

            # train and evaluate top H individuals
            tic1 = time.time()
            # get min distance between each of the remaining individuals and the training set
            # dist_list = [get_min_distance(x_train, get_sequence(get_node_encoding(architecture.layers).flatten(), architecture.connections)) for architecture in population[start_index + 1:]]
            dist_list = [get_min_distance(x_train, get_model_sequences(architecture)) for architecture in
                         population[start_index + 1:]]
            toc1 = time.time()
            print('x_train length:', len(x_train))
            print('dist_list calculation time:', toc1 - tic1, 'sec')

            # print('experiment index:', exp_repeat_index, 't=' + str(t), 'got to line 123')

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
                val_acc, test_acc, train_time = evaluator.descriptor_evaluate(d, metrics=['validation_accuracy',
                                                                                          'test_accuracy', 'time_cost'])

                architecture.fitness = val_acc

                architecture.test_acc = test_acc
                architecture.train_time = train_time

                if time == 0.0:
                    continue

                new_population.append(architecture)
                # x_train.append(get_sequence(get_node_encoding(architecture.layers).flatten(), architecture.connections))
                # y_train.append(val_acc)

                # get isomorphic sequences
                isomorphic_sequences = get_all_isomorphic_sequences(architecture)
                x_train.extend(isomorphic_sequences)
                for _ in range(len(isomorphic_sequences)):
                    y_train.append(val_acc)
                # # print('isomorphic sequences (topH) initial length:', len(isomorphic_sequences))
                # isomorphic_sequences_unique = []
                # for arr in isomorphic_sequences:
                #   if not any(np.array_equal(arr, unique_arr) for unique_arr in isomorphic_sequences_unique):
                #     isomorphic_sequences_unique.append(arr)
                # # print('isomorphic sequences (topH) length:', len(isomorphic_sequences_unique))
                # x_train.extend(isomorphic_sequences_unique)
                # for _ in range(len(isomorphic_sequences_unique)):
                #     y_train.append(val_acc)

                if best_val_acc != []:
                    if val_acc > best_val_acc[-1]:
                        best_val_acc.append(val_acc)
                        best_test_acc.append(test_acc)
                    else:
                        best_val_acc.append(best_val_acc[-1])
                        best_test_acc.append(best_test_acc[-1])
                else:
                    best_val_acc.append(val_acc)
                    best_test_acc.append(test_acc)
                train_times.append(train_time)
                if total_time != []:
                    total_time.append(total_time[-1] + train_time)
                else:
                    total_time.append(train_time)
                current_time_budget += train_time

                num_arch += 1

            with open(folder_name + '/topH_iteration' + str(num_file) + '.txt', 'w') as f:
                ind_num = num_topK
                for index in range(num_topK, len(new_population)):
                    ind = new_population[index]
                    ind_num += 1
                    f.write('architecture' + str(ind_num) + '\n')
                    f.write('layers: ')
                    for op in ind.layers:
                        f.write(op + ' ')
                    f.write('\n')
                    f.write('connections: ')
                    for conn in ind.connections:
                        f.write(str(int(conn)) + ' ')
                    f.write('\n')
                    f.write('fitness (validation accuracy): ')
                    f.write(str(ind.fitness))
                    f.write('\n')
                    f.write('test accuracy: ')
                    f.write(str(ind.test_acc))
                    f.write('\n')
                    f.write('train time: ')
                    f.write(str(ind.train_time))
                    f.write('\n')

            # update population
            if len(new_population) != 0:
                population = new_population

            # train fitness approximation
            with open(folder_name + '/xgb_stats_iteration' + str(num_file) + '.txt', 'w') as f:
                with redirect_stdout(f):
                    # xgb_model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1)
                    xgb_model = XGBRegressor(eta=0.1)
                    if t > 1:
                        xgb_model.fit(np.array(x_train), np.array(y_train), eval_set=[(x_train, y_train), (x_val, y_val)],
                                      eval_metric='rmse')
                    else:
                        xgb_model.fit(np.array(x_train), np.array(y_train), eval_set=[(x_train, y_train)],
                                      eval_metric='rmse')
                    xgb_stats = xgb_model.evals_result()
                    print(xgb_stats)

            # evolutionary algorithm
            total_population = []
            for epoch in range(NUM_GEN):
                new_population = []
                for i in range(POPULATION_SIZE):
                    individual = copy.deepcopy(tournament_selection(population))
                    new_individual = bitwise_mutation(individual)

                    # new_individual.fitness = xgb_model.predict(np.array([get_sequence(get_node_encoding(new_individual.layers).flatten(), new_individual.connections)]))[0]
                    new_individual.fitness = xgb_model.predict(np.array([get_model_sequences(new_individual)]))[0]
                    new_population.append(new_individual)
                    total_population.append(new_individual)
                population = new_population

                with open(folder_name + '/population_iteration' + str(num_file) + '_epoch' + str(epoch + 1) + '.txt',
                          'w') as f:
                    ind_num = 0
                    for ind in population:
                        ind_num += 1
                        f.write('architecture' + str(ind_num) + '\n')
                        f.write('layers: ')
                        for op in ind.layers:
                            f.write(op + ' ')
                        f.write('\n')
                        f.write('connections: ')
                        for conn in ind.connections:
                            f.write(str(int(conn)) + ' ')
                        f.write('\n')
                        f.write('fitness (validation accuracy): ')
                        f.write(str(ind.fitness))
                        f.write('\n')
                        f.write('test accuracy: ')
                        f.write(str(ind.test_acc))
                        f.write('\n')
                        f.write('train time: ')
                        f.write(str(ind.train_time))
                        f.write('\n')

            # validation set for next iteration's xgboost model
            x_val = x_train
            y_val = y_train

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for iteration t=' + str(t) + ':', toc - tic, 'sec')
            print('current time budget:', current_time_budget, 'max time budget:', MAX_TIME_BUDGET)

        end_time = time.time()

        with open(folder_name + '/best_val_acc' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_val_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_test_acc' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_test_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/train_times' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in train_times:
                f.write(str(element) + '\n')

        with open(folder_name + '/total_time' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in total_time:
                f.write(str(element) + '\n')

        with open(folder_name + '/execution_time' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            f.write(str(end_time - start_time) + '\n')  # in seconds


if __name__ == '__main__':
    NAS_EA_FA_V2()
