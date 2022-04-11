import numpy as np
from nord.neural_nets import BenchmarkEvaluator, NASWT_Evaluator
from nas_101 import ModelSpec, Network
import os
from xgboost import XGBRegressor
import copy
from contextlib import redirect_stdout
import time

from params import EXP_REPEAT_TIMES, MAX_TIME_BUDGET, POPULATION_SIZE, NUM_GEN, T, K, H
from nasbench101_utils_dnc import MAX_CONNECTIONS
from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, \
    get_all_isomorphic_sequences, get_min_distance, get_model_sequences, tournament_selection, bitwise_mutation

from save_individual import save_individual_101_dnc, save_individual_fitness_approximation

import argparse

parser = argparse.ArgumentParser(description='NASBench')
parser.add_argument('--module_vertices', default=7, type=int, help='#vertices in graph')
parser.add_argument('--max_edges', default=9, type=int, help='max edges in graph')
parser.add_argument('--available_ops', default=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                    type=list, help='available operations performed on vertex')
parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='#epochs of training')
parser.add_argument('--learning_rate', default=0.025, type=float, help='base learning rate')
parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')
parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
parser.add_argument('--num_labels', default=10, type=int, help='#classes')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')

args = parser.parse_args(args=[])


def NAS_EA_FA_V2_naswt():
    # Instantiate the evaluators
    benchmark_evaluator = BenchmarkEvaluator()
    naswt_evaluator = NASWT_Evaluator()

    if not os.path.exists('results_nas_ea_fa_v2_dnc101_naswt_' + str(args.batch_size)):
        os.mkdir('results_nas_ea_fa_v2_dnc101_naswt_' + str(args.batch_size))
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = 'results_nas_ea_fa_v2_dnc101_naswt_' + str(args.batch_size) + '/results' + str(exp_repeat_index + 1)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_score = []
        best_val_acc = []
        best_test_acc = []
        train_times = []
        naswt_calc_times = []
        total_train_time = []
        total_naswt_calc_time = []

        best_score_absolute = []
        best_val_acc_based_on_fitness = []
        best_test_acc_based_on_fitness = []

        x_train = []
        y_train = []

        current_time_budget = 0

        # Randomly sample POPULATION_SIZE architectures with an initial fitness of 0
        total_population = []
        for _ in range(POPULATION_SIZE):
            is_valid_architecture = False
            while not is_valid_architecture:
                architecture = randomly_sample_architecture()
                # check if connection number is ok for nasbench-101
                if sum(architecture.connections) <= MAX_CONNECTIONS and architecture.valid_architecture:
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
                val_acc, train_time = benchmark_evaluator.descriptor_evaluate(d, acc='validation_accuracy')
                test_acc, train_time = benchmark_evaluator.descriptor_evaluate(d, acc='test_accuracy')

                arch = ModelSpec(matrix=architecture.simplified_connection_matrix, ops=architecture.simplified_layers)
                net = Network(arch, args)
                K_matrix, score, calc_time = naswt_evaluator.net_evaluate(net=net, batch_size=args.batch_size,
                                                                           dataset=args.dataset)
                print('topK', 'num_arch:', num_arch, 'naswt_calc_time:', calc_time, 'sec')

                architecture.fitness = score
                architecture.val_acc = val_acc
                architecture.test_acc = test_acc
                architecture.train_time = train_time
                architecture.naswt_calc_time = calc_time

                if time == 0.0:
                    continue

                new_population.append(architecture)

                # get isomorphic sequences
                isomorphic_sequences = get_all_isomorphic_sequences(architecture)
                x_train.extend(isomorphic_sequences)
                for _ in range(len(isomorphic_sequences)):
                    y_train.append(score)

                if best_val_acc != []:
                    if val_acc > best_val_acc[-1]:
                        best_score.append(score)
                        best_val_acc.append(val_acc)
                        best_test_acc.append(test_acc)
                    else:
                        best_score.append(best_score[-1])
                        best_val_acc.append(best_val_acc[-1])
                        best_test_acc.append(best_test_acc[-1])
                else:
                    best_score.append(score)
                    best_val_acc.append(val_acc)
                    best_test_acc.append(test_acc)

                train_times.append(train_time)
                naswt_calc_times.append(calc_time)

                if total_train_time != []:
                    total_train_time.append(total_train_time[-1] + train_time)
                    total_naswt_calc_time.append(total_naswt_calc_time[-1] + calc_time)
                else:
                    total_train_time.append(train_time)
                    total_naswt_calc_time.append(calc_time)

                if best_score_absolute != []:
                    if score > best_score_absolute[-1]:
                        best_score_absolute.append(score)
                        best_val_acc_based_on_fitness.append(val_acc)
                        best_test_acc_based_on_fitness.append(test_acc)
                    else:
                        best_score_absolute.append(best_score_absolute[-1])
                        best_val_acc_based_on_fitness.append(best_val_acc_based_on_fitness[-1])
                        best_test_acc_based_on_fitness.append(best_test_acc_based_on_fitness[-1])
                else:
                    best_score_absolute.append(score)
                    best_val_acc_based_on_fitness.append(val_acc)
                    best_test_acc_based_on_fitness.append(test_acc)

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
                    save_individual_101_dnc(f, ind, ind_num)

            num_topK = len(new_population)

            # train and evaluate top H individuals
            tic1 = time.time()
            # get min distance between each of the remaining individuals and the training set
            # dist_list = [get_min_distance(x_train, get_sequence(get_node_encoding(architecture.layers).flatten(), architecture.connections)) for architecture in population[start_index + 1:]]
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
                val_acc, train_time = benchmark_evaluator.descriptor_evaluate(d, acc='validation_accuracy')
                test_acc, train_time = benchmark_evaluator.descriptor_evaluate(d, acc='test_accuracy')

                arch = ModelSpec(matrix=architecture.simplified_connection_matrix, ops=architecture.simplified_layers)
                net = Network(arch, args)
                K_matrix, score, calc_time = naswt_evaluator.net_evaluate(net=net, batch_size=args.batch_size,
                                                                           dataset=args.dataset)
                print('topH', 'num_arch:', num_arch, 'naswt_calc_time:', calc_time, 'sec')

                architecture.fitness = score
                architecture.val_acc = val_acc
                architecture.test_acc = test_acc
                architecture.train_time = train_time
                architecture.naswt_calc_time = calc_time

                if time == 0.0:
                    continue

                new_population.append(architecture)

                # get isomorphic sequences
                isomorphic_sequences = get_all_isomorphic_sequences(architecture)
                x_train.extend(isomorphic_sequences)
                for _ in range(len(isomorphic_sequences)):
                    y_train.append(score)

                if best_val_acc != []:
                    if val_acc > best_val_acc[-1]:
                        best_score.append(score)
                        best_val_acc.append(val_acc)
                        best_test_acc.append(test_acc)
                    else:
                        best_score.append(best_score[-1])
                        best_val_acc.append(best_val_acc[-1])
                        best_test_acc.append(best_test_acc[-1])
                else:
                    best_score.append(score)
                    best_val_acc.append(val_acc)
                    best_test_acc.append(test_acc)

                train_times.append(train_time)
                naswt_calc_times.append(calc_time)

                if total_train_time != []:
                    total_train_time.append(total_train_time[-1] + train_time)
                    total_naswt_calc_time.append(total_naswt_calc_time[-1] + calc_time)
                else:
                    total_train_time.append(train_time)
                    total_naswt_calc_time.append(calc_time)

                if best_score_absolute != []:
                    if score > best_score_absolute[-1]:
                        best_score_absolute.append(score)
                        best_val_acc_based_on_fitness.append(val_acc)
                        best_test_acc_based_on_fitness.append(test_acc)
                    else:
                        best_score_absolute.append(best_score_absolute[-1])
                        best_val_acc_based_on_fitness.append(best_val_acc_based_on_fitness[-1])
                        best_test_acc_based_on_fitness.append(best_test_acc_based_on_fitness[-1])
                else:
                    best_score_absolute.append(score)
                    best_val_acc_based_on_fitness.append(val_acc)
                    best_test_acc_based_on_fitness.append(test_acc)

                current_time_budget += train_time

                num_arch += 1

            with open(folder_name + '/topH_iteration' + str(num_file) + '.txt', 'w') as f:
                ind_num = num_topK
                for index in range(num_topK, len(new_population)):
                    ind = new_population[index]
                    ind_num += 1
                    save_individual_101_dnc(f, ind, ind_num)

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
                        save_individual_fitness_approximation(f, ind, ind_num, 'naswt_score')

            # validation set for next iteration's xgboost model
            x_val = x_train
            y_val = y_train

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for iteration t=' + str(t) + ':', toc - tic, 'sec')
            print('current time budget:', current_time_budget, 'max time budget:', MAX_TIME_BUDGET)

        end_time = time.time()

        with open(folder_name + '/best_naswt_score' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_score:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_val_acc' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_val_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_test_acc' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_test_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/train_times' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in train_times:
                f.write(str(element) + '\n')

        with open(folder_name + '/total_train_time' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in total_train_time:
                f.write(str(element) + '\n')

        with open(folder_name + '/naswt_calc_times' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in naswt_calc_times:
                f.write(str(element) + '\n')

        with open(folder_name + '/total_naswt_calc_time' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in total_naswt_calc_time:
                f.write(str(element) + '\n')

        with open(folder_name + '/execution_time' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            f.write(str(end_time - start_time) + '\n')  # in seconds

        with open(folder_name + '/best_naswt_score_absolute' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_score_absolute:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_val_acc_based_on_fitness' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_val_acc_based_on_fitness:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_test_acc_based_on_fitness' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_test_acc_based_on_fitness:
                f.write(str(element) + '\n')


if __name__ == '__main__':
    NAS_EA_FA_V2_naswt()
